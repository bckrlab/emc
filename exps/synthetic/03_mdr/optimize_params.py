import json
import os

import numpy as np
import optuna
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.metrics import cluster
from sklearn.preprocessing import OneHotEncoder

from emc.estimator.EMC import EMC
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "03_mdr", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")
optm_runs = data["optm"]

# parameters
params = {}

def emc_objective(trial):

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        lambda_f = trial.suggest_float("lambda_f", 0.92, 0.95, step=0.01)
        lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99, step=0.01)
        beta = trial.suggest_float("beta", 0, 0)
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
            alpha=4,
            order=1,
            lambda_=[lambda_f, lambda_s],
            beta=beta,
            delta=[delta_f, delta_s],
            eta=[eta_f, eta_s],
            tau=tau
        )
        emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)

        # evaluate
        labels_true = run_params["subprocess_sequence_full"]
        labels_pred = emc_ins.pred_mode_hist
        emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
        ari_vec.append(emc_ari)

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ari_vec)

def clustream_objective(trial):

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        n_macro_clusters = trial.suggest_int("n_macro_clusters", 5, 10)
        max_micro_clusters = trial.suggest_int("max_micro_clusters", 50, 150)
        micro_cluster_r_factor = trial.suggest_int("micro_cluster_r_factor", 2, 10)
        time_window = trial.suggest_int("time_window", 500, 1500)
        time_gap = trial.suggest_int("time_gap", 50, 150)

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        # run
        X = [[x] for x in run_params["symbol_sequence"]]
        enc = OneHotEncoder(handle_unknown="ignore")
        X_enc = enc.fit_transform(X).toarray()
        labels_pred_clustream = []
        clustream = CluStream(
            n_macro_clusters=n_macro_clusters,
            max_micro_clusters=max_micro_clusters,
            micro_cluster_r_factor=micro_cluster_r_factor,
            time_window=time_window,
            time_gap=time_gap,
            seed=42,
        )
        for i, (x, _) in enumerate(iter_array(X_enc)):
            clustream.learn_one(x)
            if i > 20:
                labels_pred_clustream.append(clustream.predict_one(x))
            # if i % 1000 == 0: print(f"step: {step}, i: {i}/{len(X_enc)}")

        # evaluate
        labels_true = run_params["subprocess_sequence_full"]
        length_diff = len(labels_true) - len(labels_pred_clustream)
        clustream_ari = cluster.adjusted_rand_score(
            labels_pred=labels_pred_clustream,
            labels_true=labels_true[length_diff:]
        )

        ari_vec.append(clustream_ari)

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ari_vec)

def dbstream_objective(trial):

    ari_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        clustering_threshold = trial.suggest_float("clustering_threshold", 1, 10)
        fading_factor = trial.suggest_float("fading_factor", 0.001, 0.1)
        cleanup_interval = trial.suggest_int("cleanup_interval", 1, 20)
        intersection_factor = trial.suggest_float("intersection_factor", 0.03, 0.8)
        minimum_weight = trial.suggest_float("minimum_weight", 0.1, 10)

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        # run
        X = [[x] for x in run_params["symbol_sequence"]]
        enc = OneHotEncoder(handle_unknown="ignore")
        X_enc = enc.fit_transform(X).toarray()
        labels_pred_dbstream = []
        dbstream = DBSTREAM(
            clustering_threshold=clustering_threshold,
            fading_factor=fading_factor,
            cleanup_interval=cleanup_interval,
            intersection_factor=intersection_factor,
            minimum_weight=minimum_weight
        )
        for i, (x, _) in enumerate(iter_array(X_enc)):
            dbstream.learn_one(x)
            labels_pred_dbstream.append(dbstream.predict_one(x))

        # evaluate
        labels_true = run_params["subprocess_sequence_full"]
        length_diff = len(labels_true) - len(labels_pred_dbstream)
        dbstream_ari = cluster.adjusted_rand_score(
            labels_pred=labels_pred_dbstream,
            labels_true=labels_true[length_diff:]
        )
        ari_vec.append(dbstream_ari)

        # prune
        intermediate_value = np.mean(ari_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(ari_vec)

# run optimization for each algorithm
algs = {
    "emc": emc_objective,
    "clustream": clustream_objective,
    "dbstream": dbstream_objective
}
for alg, objective in algs.items():
    print(f"Optimizing {alg} parameters...")
    study = optuna.create_study(
        study_name=alg,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.SuccessiveHalvingPruner()
    )
    optuna.logging.set_verbosity(optuna.logging.WARN)
    if alg == "clustream":
        study.enqueue_trial({ # start with default parameters
            "n_macro_clusters": 5,
            "max_micro_clusters": 100,
            "micro_cluster_r_factor": 2,
            "time_window": 1000,
            "time_gap": 100
        })
    study.optimize(objective, n_trials=1500, timeout=60*60*12, show_progress_bar=True)
    params[alg] = study.best_params

# save parameters
params_path = os.path.join(output_dir_path, "params.json")
with open(params_path, "w") as fp:
    json.dump(params, fp)
print(f"params saved: {params_path}")
