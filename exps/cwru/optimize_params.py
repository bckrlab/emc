import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import cluster
from tqdm.autonotebook import tqdm

from emc.estimator.EMC import EMC
from emc.utils.data import grouped_decompose
from emc.utils.loader import load_cwru_data, load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "cwru", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")

# load alphabet
alphabet_path = os.path.join(alphabet_dir_path, "ac:27_pl:2.pkl")
alphabet = load_pickle_file(alphabet_path)
if alphabet is not None: 
    print(f"{alphabet_path} loaded")
    prim_len = len(alphabet.cluster_centers_[0])
    print(f"  primitive length: {prim_len}")
    alp_car = len(alphabet.cluster_centers_)
    print(f"  alphabet cardinality: {alp_car}")

# load data
X_opt, cp_opt, y_true_opt = load_cwru_data(condition_sequence=["L0-IR:07","L0-IR:28"], points_per_condition=12000)

# discretize
symbol_sequence_opt = alphabet.predict(grouped_decompose(X_opt, prim_len))

# params
params = {}

# run optimization
def objective(trial):

    # get parameters
    order = trial.suggest_int("order", 2, 2, step=1)
    lambda_f = trial.suggest_float("lambda_f", 0.92, 0.95)
    lambda_s = trial.suggest_float("lambda_s", 0.96, 0.98)
    beta = trial.suggest_float("beta", 0, 0.01)
    delta_f = trial.suggest_float("delta_f", 0.05, 0.5)
    delta_s = trial.suggest_float("delta_s", 0.05, 0.5)
    eta_f = trial.suggest_float("eta_f", 0.05, 0.5)
    eta_s = trial.suggest_float("eta_s", 0.05, 0.5)
    tau = trial.suggest_int("tau", 20, 100)

    # avoid duplicate trials
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
    
    # run
    emc_ins = EMC(
        alpha=alp_car,
        order=order,
        lambda_=[lambda_f, lambda_s],
        beta=beta,
        delta=[delta_f, delta_s],
        eta=[eta_f, eta_s],
        tau=tau
    )
    emc_ins.process_sequence(symbol_sequence_opt, progress=False)

    # evaluate
    labels_true = np.array(y_true_opt)
    labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
    length_diff = len(labels_true) - len(labels_pred)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    return emc_ari

# run optuna
study = optuna.create_study(
    study_name="emc_cwru",
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
