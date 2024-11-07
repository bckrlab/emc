import json
import os

import numpy as np
import optuna

from emc.estimator.EMC import EMC
from emc.estimator.MC_ADWIN import MC_ADWIN
from emc.estimator.MC_SW import MC_SW
from emc.utils.evaluator import evaluate_estimates
from emc.utils.loader import load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "01_pt", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")
optm_runs = data["optm"]

# parameters
params = {}

def emc_objective(trial):

    mae_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        lambda_f = trial.suggest_float("lambda_f", 0.90, 0.95, step=0.01)
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
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            lambda_=[lambda_f, lambda_s],
            beta=beta,
            delta=[delta_f, delta_s],
            eta=[eta_f, eta_s],
            tau=tau,
        )
        emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)

        # evaluate
        true_matrices = [run_params["subprocesses"][sid].transition_matrix for sid in run_params["subprocess_sequence"]]
        emc_mae, emc_ae = evaluate_estimates(
            estimates=emc_ins.P_exp_hist,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=emc_ins.index_symbol_map
        )
        mae_vec.append(emc_mae)

        # prune
        intermediate_value = np.mean(mae_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_vec)

def mcadwin_objective(trial):

    mae_vec = []
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
        
        # run
        mc_adwin_ins = MC_ADWIN(
            alpha=data["meta"]["alp_car"],
            order=data["meta"]["k"],
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
            grace_period=grace_period,
        )
        mc_adwin_ins.process_sequence(run_params["symbol_sequence"])

        # evaluate
        true_matrices = [run_params["subprocesses"][sid].transition_matrix for sid in run_params["subprocess_sequence"]]
        mc_adwin_mae, mc_adwin_ae = evaluate_estimates(
            estimates=mc_adwin_ins.estimates,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=mc_adwin_ins.index_symbol_map
        )
        mae_vec.append(mc_adwin_mae)

        # prune
        intermediate_value = np.mean(mae_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_vec)

def mcsw_objective(trial):

    mae_vec = []
    for step, (run_id, run_params) in enumerate(optm_runs.items()):

        # get parameters
        window_size = trial.suggest_int("window_size", 250, 750)

        # avoid duplicate trials
        trials_to_consider = trial.study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        for t in reversed(trials_to_consider):
            if trial.params == t.params:
                return t.value
        
        # run
        mc_sw_ins = MC_SW(
            order=data["meta"]["k"],
            alpha=data["meta"]["alp_car"],
            window_size=window_size
        )
        mc_sw_ins.process_sequence(run_params["symbol_sequence"])
        
        # evaluate
        true_matrices = [run_params["subprocesses"][sid].transition_matrix for sid in run_params["subprocess_sequence"]]
        mc_sw_mae, mc_sw_ae = evaluate_estimates(
            estimates=mc_sw_ins.estimates,
            true_matrices=true_matrices,
            regime_lengths=run_params["regime_lengths"],
            index_symbol_map=mc_sw_ins.index_symbol_map
        )
        mae_vec.append(mc_sw_mae)

        # prune
        intermediate_value = np.mean(mae_vec)
        trial.report(intermediate_value, step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return np.mean(mae_vec)

# run optimization for each algorithm
algs = {
    "emc": emc_objective,
    "mc_adwin": mcadwin_objective,
    "mc_sw": mcsw_objective
}
for alg, objective in algs.items():
    print(f"Optimizing {alg} parameters...")
    study = optuna.create_study(
        study_name=alg,
        direction="minimize",
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
