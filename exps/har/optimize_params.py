import json

import numpy as np
import optuna
import os
import pandas as pd
from sklearn.metrics import cluster
from tqdm.auto import tqdm

from emc.estimator.EMC import EMC
from emc.utils.data import grouped_decompose
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "har", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")

# load alphabet
alphabet_id = "ac:7_pl:5"
alphabet_path = os.path.join(alphabet_dir_path, f"{alphabet_id}.pkl")
alphabet = load_pickle_file(alphabet_path)
if alphabet is not None: 
    print(f"{alphabet_path} loaded")
    prim_len = int(len(alphabet.cluster_centers_[0])/3)
    print(f"  primitive length: {prim_len}")
    alp_car = len(alphabet.cluster_centers_)
    print(f"  alphabet cardinality: {alp_car}")

# load scaler
scaler_path = os.path.join(alphabet_dir_path, f"{alphabet_id}_sc.pkl")
scaler = load_pickle_file(scaler_path)
if scaler is not None: 
    print(f"\n{scaler_path} loaded")

# params
params = {}

# load data
act_lbl_idx_map = {
    "climbingdown": 1,
    "climbingup": 2,
    "jumping": 3,
    "lying": 4,
    "running": 5,
    "sitting": 6,
    "standing": 7,
    "walking": 8
}
act_idx_lbl_map = {y: x for x, y in act_lbl_idx_map.items()}
sensor = "chest"
subjects_optim = [1]
activities = [1,2,3,4,5,6,7,8]
number_of_runs = 10
number_of_regimes = 10
master_rng = np.random.default_rng(42)
local_seeds = master_rng.integers(low=0, high=999999, size=number_of_runs, dtype=int)
optm_runs = {}
pbar_conf = {"total": len(subjects_optim)*number_of_runs}
with tqdm(**pbar_conf) as pbar:
    for run_index in range(number_of_runs):
        rng = np.random.default_rng(local_seeds[run_index])
        # generate a random sequence of activities
        while True:
            rnd_act_idx_seq = rng.choice(a=activities, size=number_of_regimes)
            if 0 not in np.diff(rnd_act_idx_seq):
                break
        for subject in subjects_optim:
            # combine data of randomly generated activity sequence
            rnd_act_lbl_seq = [act_idx_lbl_map[a] for a in rnd_act_idx_seq]
            X_seq = pd.DataFrame()
            regime_lengths = []
            for act_lbl in rnd_act_lbl_seq:
                data_path = os.path.join(paths["data"]["realworld_har"]["samples"], f"subject_{subject}", act_lbl, f"{sensor}.csv")
                data = pd.read_csv(data_path, sep=",", header=0, usecols=["attr_x","attr_y","attr_z"])
                X_seq = pd.concat([X_seq, data], ignore_index=True)
                regime_lengths.append(len(data))
            X_sc_opt = scaler.transform(X_seq)
            grouped_target_signal = grouped_decompose(X_sc_opt, prim_len)
            prim_id_seq = alphabet.predict(grouped_target_signal)
            optm_runs[f"s{subject}_r{run_index+1}"] = {
                "subprocess_sequence": rnd_act_idx_seq,
                "regime_lengths": regime_lengths,
                "change_points": np.insert(np.cumsum(regime_lengths), 0, 0),
                "discrete_sequence": prim_id_seq
            }
            pbar.update()

def objective(trial):

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        lambda_f = trial.suggest_float("lambda_f", 0.92, 0.95, step=0.01)
        lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99, step=0.01)
        beta = trial.suggest_float("beta", 0, 0.05, step=0.005)
        delta_f = trial.suggest_float("delta_f", 0.05, 0.5, step=0.05)
        delta_s = trial.suggest_float("delta_s", 0.05, 0.5, step=0.05)
        eta_f = trial.suggest_float("eta_f", 0.05, 0.5, step=0.05)
        eta_s = trial.suggest_float("eta_s", 0.05, 0.5, step=0.05)
        tau = trial.suggest_int("tau", 25, 100, step=25)

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value

        # run
        emc_ins = EMC(
            alpha=alp_car,
            order=1,
            lambda_=[lambda_f, lambda_s],
            beta=beta,
            delta=[delta_f, delta_s],
            eta=[eta_f, eta_s],
            tau=tau
        )
        emc_ins.process_sequence(run_params["discrete_sequence"], progress=False)

        # evaluate
        labels_true = np.repeat(run_params["subprocess_sequence"], run_params["regime_lengths"])
        labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
        length_diff = len(labels_true) - len(labels_pred)
        if length_diff > 0:
            labels_true = labels_true[:-length_diff]
        emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        ari_vec.append(emc_ari)

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ari_vec)

# run optuna
study = optuna.create_study(
    study_name="emc_har",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.SuccessiveHalvingPruner()
)
optuna.logging.set_verbosity(optuna.logging.WARN)
study.optimize(objective, n_trials=250, timeout=None, show_progress_bar=True)
params["emc"] = study.best_params

# save parameters
params_path = os.path.join(output_dir_path, "params.json")
with open(params_path, "w") as fp:
    json.dump(params, fp)
print(f"params saved: {params_path}")
