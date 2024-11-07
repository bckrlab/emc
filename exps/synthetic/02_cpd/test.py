import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from river.drift import ADWIN, KSWIN, PageHinkley
from tqdm.autonotebook import tqdm

from emc.estimator.EMC import EMC
from emc.utils.evaluator import evaluate_cpd
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "02_cpd", "output")

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"{params_path} loaded")
    # print(params)

def execute_single_run(run_desc):

    run_id, run_params = run_desc
    result_dict = {"run_id": run_id}
    true_cps = run_params["change_points"][:-1]
    sequence_length = len(run_params["symbol_sequence"])

    # EMC
    emc_ins = EMC(
        alpha=data["meta"]["alp_car"],
        order=data["meta"]["k"],
        lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
        beta=params["emc"]["beta"],
        delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
        eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
        tau=params["emc"]["tau"],
    )
    emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)
    emc_cpd_result = evaluate_cpd(
        true_cps=true_cps,
        detected_cps=np.array(emc_ins.detected_changes),
        margin_of_error=data["meta"]["margin_of_error"],
        allow_prior=False,
        sequence_length=sequence_length
    )
    result_dict["emc:cpd_tp"] = emc_cpd_result["tp"]
    result_dict["emc:cpd_fp"] = emc_cpd_result["fp"]
    result_dict["emc:cpd_tn"] = emc_cpd_result["tn"]
    result_dict["emc:cpd_fn"] = emc_cpd_result["fn"]
    result_dict["emc:cpd_f1"] = emc_cpd_result["f1"]
    result_dict["emc:cpd_mdl"] = emc_cpd_result["mdl"]

    # ADWIN
    adwin_ins = ADWIN(
        delta=params["adwin"]["delta"],
        clock=params["adwin"]["clock"],
        max_buckets=params["adwin"]["max_buckets"],
        min_window_length=params["adwin"]["min_window_length"]
    )
    detected_changes = []
    for i, symbol in enumerate(run_params["symbol_sequence"]):
        _ = adwin_ins.update(symbol)
        if adwin_ins.drift_detected:
            detected_changes.append(i)
    adwin_cpd_result = evaluate_cpd(
        true_cps=true_cps,
        detected_cps=detected_changes,
        margin_of_error=data["meta"]["margin_of_error"],
        allow_prior=False,
        sequence_length=sequence_length
    )
    result_dict["adwin:cpd_tp"] = adwin_cpd_result["tp"]
    result_dict["adwin:cpd_fp"] = adwin_cpd_result["fp"]
    result_dict["adwin:cpd_tn"] = adwin_cpd_result["tn"]
    result_dict["adwin:cpd_fn"] = adwin_cpd_result["fn"]
    result_dict["adwin:cpd_f1"] = adwin_cpd_result["f1"]
    result_dict["adwin:cpd_mdl"] = adwin_cpd_result["mdl"]

    # KSWIN
    kswin_ins = KSWIN(
        alpha=params["kswin"]["alpha"],
        window_size=params["kswin"]["window_size"],
        stat_size=params["kswin"]["stat_size"],
        seed=42
    )
    detected_changes = []
    for i, symbol in enumerate(run_params["symbol_sequence"]):
        _ = kswin_ins.update(symbol)
        if kswin_ins.drift_detected:
            detected_changes.append(i)
    kswin_cpd_result = evaluate_cpd(
        true_cps=true_cps,
        detected_cps=detected_changes,
        margin_of_error=data["meta"]["margin_of_error"],
        allow_prior=False,
        sequence_length=sequence_length
    )
    result_dict["kswin:cpd_tp"] = kswin_cpd_result["tp"]
    result_dict["kswin:cpd_fp"] = kswin_cpd_result["fp"]
    result_dict["kswin:cpd_tn"] = kswin_cpd_result["tn"]
    result_dict["kswin:cpd_fn"] = kswin_cpd_result["fn"]
    result_dict["kswin:cpd_f1"] = kswin_cpd_result["f1"]
    result_dict["kswin:cpd_mdl"] = kswin_cpd_result["mdl"]

    # PHT
    pht_ins = PageHinkley(
        min_instances=params["pht"]["min_instances"],
        delta=params["pht"]["delta"],
        threshold=params["pht"]["threshold"],
        alpha=params["pht"]["alpha"],
        mode=params["pht"]["mode"]
    )
    detected_changes = []
    for i, symbol in enumerate(run_params["symbol_sequence"]):
        _ = pht_ins.update(symbol)
        if pht_ins.drift_detected:
            detected_changes.append(i)
    pht_cpd_result = evaluate_cpd(
        true_cps=true_cps,
        detected_cps=detected_changes,
        margin_of_error=data["meta"]["margin_of_error"],
        allow_prior=False,
        sequence_length=sequence_length
    )
    result_dict["pht:cpd_tp"] = pht_cpd_result["tp"]
    result_dict["pht:cpd_fp"] = pht_cpd_result["fp"]
    result_dict["pht:cpd_tn"] = pht_cpd_result["tn"]
    result_dict["pht:cpd_fn"] = pht_cpd_result["fn"]
    result_dict["pht:cpd_f1"] = pht_cpd_result["f1"]
    result_dict["pht:cpd_mdl"] = pht_cpd_result["mdl"]

    return result_dict

# run tasks in parallel
test_runs = data["test"]
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=64) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        results.append(result)

# format results table
results_df = pd.DataFrame.from_records(results)
results_df_frm = results_df.drop(columns=["run_id"])
results_df_frm = results_df_frm\
    .agg(lambda x: fr"${x.mean():.2f}\pm{x.std():.2f}$")\
    .transpose().reset_index()
results_df_frm = results_df_frm.rename(columns={"index": "alg"})
results_df_frm[['Prefix', 'Suffix']] = results_df_frm['alg'].str.split(':cpd_', n=1, expand=True)
results_df_frm = results_df_frm.pivot(index='Prefix', columns='Suffix').reset_index()
results_df_frm = results_df_frm.drop(columns=["alg"], level=0)
results_df_frm = results_df_frm.droplevel(axis=1, level=0)
results_df_frm = results_df_frm.rename(columns={"": "alg"})
results_df_frm["alg"] = results_df_frm["alg"].apply(lambda x: x.upper().replace("_"," "))
results_df_frm = results_df_frm[["alg","f1","fn","fp","mdl"]]

# save results
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)
results_path = os.path.join(results_dir_path, "syn_cpd_f1.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.3f")
print(f"results saved: {results_path}")
