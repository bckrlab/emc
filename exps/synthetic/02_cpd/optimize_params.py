import json
import os

import numpy as np
import optuna
from river.drift import ADWIN, KSWIN, PageHinkley

from emc.estimator.EMC import EMC
from emc.utils.evaluator import evaluate_cpd
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "02_cpd", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")
optm_runs = data["optm"]

# parameters
params = {}

def emc_objective(trial):

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

       # get parameters
        lambda_f = trial.suggest_float("lambda_f", 0.90, 0.95, step=0.01)
        lambda_s = trial.suggest_float("lambda_s", 0.95, 0.99, step=0.01)
        beta = trial.suggest_float("beta", 0, 0, step=0.01)
        delta_f = trial.suggest_float("delta_f", 0.05, 0.5, step=0.05)
        delta_s = trial.suggest_float("delta_s", 0.05, 0.5, step=0.05)
        eta_f = trial.suggest_float("eta_f", 0.05, 0.5, step=0.05)
        eta_s = trial.suggest_float("eta_s", 0.05, 0.5, step=0.05)
        tau = trial.suggest_int("tau", 50, 100, step=25)

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        # run
        emc_ins = EMC(
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            lambda_=[lambda_f, lambda_s],
            beta=beta,
            delta=[delta_f, delta_s],
            eta=[eta_f, eta_s],
            tau=tau,
        )
        emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)

        emc_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=np.array(emc_ins.detected_changes),
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(emc_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def adwin_objective(trial):

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        delta = trial.suggest_float("delta", 0.0002, 0.02)
        clock = trial.suggest_int("clock", 3, 320)
        max_buckets = trial.suggest_int("max_buckets", 1, 50)
        min_window_length = trial.suggest_int("min_window_length", 1, 50)
        grace_period = trial.suggest_int("grace_period", 1, 100)

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        adwin_ins = ADWIN(
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period
        )
        detected_changes = []
        for i, symbol in enumerate(run_params["symbol_sequence"]):
            _ = adwin_ins.update(symbol)
            if adwin_ins.drift_detected:
                detected_changes.append(i)
        adwin_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=detected_changes,
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(adwin_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def kswin_objective(trial):

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        alpha = trial.suggest_float("alpha", 0.0005, 0.05)
        window_size = trial.suggest_int("window_size", 10, 1000)
        stat_size = trial.suggest_int("stat_size", 3, 300)

        # stat_size must be smaller than window_size
        if not stat_size < window_size:
            raise optuna.TrialPruned()
        
        # Sample larger than population or is negative:
        # self._rng.sample(range(self.window_size - self.stat_size), self.stat_size)
        if not window_size-stat_size >= stat_size:
            raise optuna.TrialPruned()

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        kswin_ins = KSWIN(
            alpha=alpha,
            window_size=window_size,
            stat_size=stat_size,
            seed=42
        )
        detected_changes = []
        for i, symbol in enumerate(run_params["symbol_sequence"]):
            _ = kswin_ins.update(symbol)
            if kswin_ins.drift_detected:
                detected_changes.append(i)
        kswin_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=detected_changes,
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(kswin_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

def pht_objective(trial):

    f1_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        min_instances = trial.suggest_int("min_instances", 3, 300)
        delta = trial.suggest_float("delta", 0.0005, 0.05)
        threshold = trial.suggest_float("threshold", 5, 500)
        alpha = trial.suggest_float("alpha", 0.9, 0.99999)
        mode = trial.suggest_categorical("mode", ["up", "down", "both"])

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        pht_ins = PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
            alpha=alpha,
            mode=mode
        )
        detected_changes = []
        for i, symbol in enumerate(run_params["symbol_sequence"]):
            _ = pht_ins.update(symbol)
            if pht_ins.drift_detected:
                detected_changes.append(i)
        pht_cpd_result = evaluate_cpd(
            true_cps=run_params["change_points"][:-1],
            detected_cps=detected_changes,
            margin_of_error=data["meta"]["margin_of_error"],
            allow_prior=False,
            sequence_length=len(run_params["symbol_sequence"])
        )
        f1_vec.append(pht_cpd_result["f1"])

        # prune
        intermediate_value = np.mean(f1_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(f1_vec)

# run optimization for each algorithm
algs = {
    "emc": emc_objective,
    "adwin": adwin_objective,
    "kswin": kswin_objective,
    "pht": pht_objective
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
    study.optimize(objective, n_trials=100, timeout=None, show_progress_bar=True)
    params[alg] = study.best_params

# save parameters
params_path = os.path.join(output_dir_path, "params.json")
with open(params_path, "w") as fp:
    json.dump(params, fp)
print(f"params saved: {params_path}")