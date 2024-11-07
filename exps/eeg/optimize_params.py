import json
import os

import numpy as np
import optuna
from sklearn.metrics import cluster

from emc.estimator.EMC import EMC
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "eeg", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")

# parameters
params = {}

# run optimization
opt_index = 3000
def objective(trial):

    # get parameters
    order = trial.suggest_int("order", 1, 1, step=1)
    lambda_f = trial.suggest_float("lambda_f", 0.92, 0.95, step=0.01)
    lambda_s = trial.suggest_float("lambda_s", 0.96, 0.99, step=0.01)
    beta = trial.suggest_float("beta", 0, 0.05, step=0.001)
    delta_f = trial.suggest_float("delta_f", 0.05, 0.5, step=0.05)
    delta_s = trial.suggest_float("delta_s", 0.05, 0.5, step=0.05)
    eta_f = trial.suggest_float("eta_f", 0.05, 0.5, step=0.05)
    eta_s = trial.suggest_float("eta_s", 0.05, 0.5, step=0.05)
    tau = trial.suggest_int("tau", 25, 150, step=25)

    # avoid duplicate trials
    trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
    for t in reversed(trials_to_consider):
        if trial.params == t.params:
            return t.value
    
    # run
    emc_ins = EMC(
        alpha=data["meta"]["n_clusters"],
        order=order,
        lambda_=[lambda_f, lambda_s],
        beta=beta,
        delta=[delta_f, delta_s],
        eta=[eta_f, eta_s],
        tau=tau
    )
    emc_ins.process_sequence(data["ms_seq"][:opt_index], progress=False)

    # ARI
    labels_true = data["Y"][:opt_index]
    labels_pred = np.array(emc_ins.pred_mode_hist)
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

    return emc_ari

# run optuna
study = optuna.create_study(
    study_name="emc_eeg",
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=optuna.pruners.SuccessiveHalvingPruner()
)
optuna.logging.set_verbosity(optuna.logging.WARN)
study.optimize(objective, n_trials=10000, timeout=None, show_progress_bar=True)
params["emc"] = study.best_params

# save parameters
params_path = os.path.join(output_dir_path, "params.json")
with open(params_path, "w") as fp:
    json.dump(params, fp)
print(f"params saved: {params_path}")
