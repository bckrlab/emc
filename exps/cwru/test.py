import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsort_keygen
from sklearn.metrics import cluster
from tqdm.autonotebook import tqdm

from emc.estimator.EMC import EMC
from emc.utils.data import grouped_decompose
from emc.utils.loader import load_cwru_data, load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "cwru", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)

# load alphabet
alphabet_path = os.path.join(alphabet_dir_path, "ac:27_pl:2.pkl")
alphabet = load_pickle_file(alphabet_path)
if alphabet is not None:
    print(f"{alphabet_path} loaded.")
    prim_len = len(alphabet.cluster_centers_[0])
    print(f"primitive length: {prim_len}")
    alp_car = len(alphabet.cluster_centers_)
    print(f"alphabet cardinality: {alp_car}")

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"\n{params_path} loaded.")
    # print(params)

# prepare test runs
test_runs = {}
sample_cnd_seqs = [
    # LOAD 0
    ["L0-OK","L0-IR:07"],
    ["L0-OK","L0-IR:14"],
    ["L0-OK","L0-IR:21"],
    ["L0-OK","L0-IR:28"],
    ["L0-OK","L0-IR:07","L0-IR:14"],
    ["L0-OK","L0-IR:14","L0-IR:21"],
    ["L0-OK","L0-IR:21","L0-IR:28"],
    # LOAD 1
    ["L1-OK","L1-IR:07"],
    ["L1-OK","L1-IR:14"],
    ["L1-OK","L1-IR:21"],
    ["L1-OK","L1-IR:28"],
    ["L1-OK","L1-IR:07","L1-IR:14"],
    ["L1-OK","L1-IR:14","L1-IR:21"],
    ["L1-OK","L1-IR:21","L1-IR:28"],
    # LOAD 2
    ["L2-OK","L2-IR:07"],
    ["L2-OK","L2-IR:14"],
    ["L2-OK","L2-IR:21"],
    ["L2-OK","L2-IR:28"],
    ["L2-OK","L2-IR:07","L2-IR:14"],
    ["L2-OK","L2-IR:14","L2-IR:21"],
    ["L2-OK","L2-IR:21","L2-IR:28"],
    # LOAD 3
    ["L3-OK","L3-IR:07"],
    ["L3-OK","L3-IR:14"],
    ["L3-OK","L3-IR:21"],
    ["L3-OK","L3-IR:28"],
    ["L3-OK","L3-IR:07","L3-IR:14"],
    ["L3-OK","L3-IR:14","L3-IR:21"],
    ["L3-OK","L3-IR:21","L3-IR:28"],
]

for sample_cnd_seq in tqdm(sample_cnd_seqs):

    # load data
    X_tst, cp_tst, y_true_tst = load_cwru_data(condition_sequence=sample_cnd_seq, points_per_condition=12000)

    # reconstruct
    symbol_sequence_test = alphabet.predict(grouped_decompose(X_tst, prim_len))

    # save test run
    test_runs[f"{sample_cnd_seq}"] = {
        "X": X_tst,
        "symbol_sequence": symbol_sequence_test,
        "change_points": cp_tst,
        "labels_true": y_true_tst,
        "sample_cnd_seq": sample_cnd_seq
    }

# execute test runs
def execute_single_run(run_desc):

    run_id, run_params = run_desc
    run_id_san = run_id.replace("[","").replace("]","").replace("'","").replace(", ","→")
    # print(run_id_san)

    result = {
        "run_id": run_id_san,
        "load": run_params["sample_cnd_seq"][0][1], # for table
        "floc": run_params["sample_cnd_seq"][1][3:5],
        "num_regs": run_id_san.count("→")+1, # used for sorting rows
    }

    # EMC
    emc_ins = EMC(
        alpha=alp_car,
        order=params["emc"]["order"],
        lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
        beta=params["emc"]["beta"],
        delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
        eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
        tau=params["emc"]["tau"],
    )
    emc_ins.process_sequence(run_params["symbol_sequence"], progress=False)

    labels_true = np.array(run_params["labels_true"])
    labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
    length_diff = len(labels_true) - len(labels_pred)
    if length_diff > 0:
        print(length_diff)
        labels_true = labels_true[:-length_diff]
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
    result["ari"] = emc_ari

    stationarity_history = np.repeat(emc_ins.stationarity_history, prim_len)
    emc_ari_s = cluster.adjusted_rand_score(
        labels_pred=labels_pred[np.argwhere(stationarity_history)].flatten(),
        labels_true=labels_true[np.argwhere(stationarity_history)].flatten()
    )
    result["ari_s"] = emc_ari_s
    result["drift_ratio"] = np.count_nonzero(stationarity_history==0)/len(stationarity_history)

    # save figure
    R = len(run_params["sample_cnd_seq"]) # number of modes
    mode_id_to_label_map = {k:run_params["sample_cnd_seq"][k].replace("IR","FD") for k in range(0,R)}
    plot_mode_transition(
        subplots=["data", "discovered_modes"],
        labels_true=labels_true,
        data=run_params["X"],
        deviation_history=emc_ins.deviation_history,
        labels_pred=labels_pred,
        change_points=run_params["change_points"],
        cp_scale_coeff=emc_ins.tau * prim_len,
        data_label="Vibration",
        data_yticks=None,
        data_include_modes=True,
        legend_text=f"ARI: {emc_ari:.2f}",
        legend_loc="upper left",
        mode_id_to_label_map=mode_id_to_label_map,
        xlabel="Time (s)",
        xticklabel_coeff=(1/12000),
        xtick_rotation=0,
        xtick_ids_to_pad=[],
        xtick_ids_to_skip=[],
        mpl_conf_path=get_mpl_conf_path("pub"),
        mpl_conf_override=None,
        fig_size=set_size(**{"width":"tpami", "aspect_ratio": 1.75, "fraction":0.5}),
        fig_path=os.path.join(results_dir_path, f"{run_id_san}.pdf"),
    )

    return result

# run tasks in parallel
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=32) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        results.append(result)

# format results
results_df = pd.DataFrame.from_records(results)
results_df_frm = results_df.copy()
results_df_frm["load"] = results_df_frm["load"].map(lambda x: f"{x} HP")
results_df_frm = results_df_frm.sort_values(by=["load","num_regs","run_id"], key=natsort_keygen())
results_df_frm["run_id"] = results_df_frm["run_id"].str.replace("→"," → ")
results_df_frm["run_id"] = results_df_frm["run_id"].str.replace("IR","FD")
results_df_frm["run_id"] = results_df_frm["run_id"].str.replace("L.-","", regex=True)
results_df_frm = results_df_frm[["run_id","load","ari","ari_s","drift_ratio"]]
results_df_frm.loc["Avg"] = results_df_frm.mean(numeric_only=True)
results_df_frm.loc["Avg","run_id"] = "Avg."

# save results
results_path = os.path.join(results_dir_path, "cwru_ari.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.2f")
print(f"results saved: {results_path}")
