import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from river.cluster import DBSTREAM, CluStream
from river.stream import iter_array
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import cluster
from sklearn.preprocessing import OneHotEncoder
from tqdm.auto import tqdm

from emc.estimator.EMC import EMC
from emc.estimator.EPSTM import construct_pst, match_pst_pairs
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "03_mdr", "output")

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
    labels_true = run_params["subprocess_sequence_full"]

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
    emc_ari = cluster.adjusted_rand_score(labels_pred=emc_ins.pred_mode_hist, labels_true=labels_true)
    result_dict["emc:dev_hist"] = emc_ins.deviation_history
    result_dict["emc:st_hist"] = emc_ins.stationarity_history
    result_dict["emc:labels_pred"] = emc_ins.pred_mode_hist
    result_dict["emc:ari"] = emc_ari

    # EPSTM (with true change points)
    psts = []
    pid_list = set(run_params["symbol_sequence"])
    cursor = 0
    for cp_idx, cp in enumerate(run_params["change_points"]):
        start = cursor
        end = cp
        pst = construct_pst(
            pst_id=f"{run_id}:reg_{start}_{end}",
            pid_list=pid_list,
            pid_seq=run_params["symbol_sequence"][start:end],
            subsequence_minimum_occurrence=2,
            subsequence_length_limit=1,
            smoothing_min_p=0.001,
            to_render=False,
            to_save=False,
            output_dir_path=None,
            log_level="DEBUG"
        )
        psts.append(pst)
        cursor = end
    dist_pairs, dist_matrix = match_pst_pairs(
        pst_list=psts,
        matching_function="epstm_2020",
        matching_parameters={"I": 0.5}
    )
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.1,
        metric="precomputed",
        linkage="single"
    )
    clusterer.fit(dist_matrix)
    epstm_tcp_labels_pred = []
    for regime_idx, regime_length in enumerate(np.diff(run_params["change_points"], prepend=0)):
        epstm_tcp_labels_pred.extend(np.repeat(clusterer.labels_[regime_idx], regime_length))
    epstm_tcp_ari = cluster.adjusted_rand_score(labels_pred=epstm_tcp_labels_pred, labels_true=labels_true)
    result_dict["epstm_tcp:ari"] = epstm_tcp_ari

    # EPSTM (with imperfect change points)
    delay_low, delay_high = [70, 160]
    fp, fn = [1, 0]
    rng = np.random.default_rng(int(run_id.split("_")[1]))
    imperfect_cps = []
    cursor = 0
    for true_cp_idx, true_cp in enumerate(run_params["change_points"]):
        if true_cp_idx == len(run_params["regime_lengths"])-1:
            delay = 0 # no delay in the final change point
        else:
            delay = rng.integers(low=delay_low, high=delay_high, endpoint=True)
        start = cursor
        end = true_cp + delay
        cursor = end
        imperfect_cps.append(end)
    if fp > 0:
        imperfect_cps.extend(rng.integers(len(run_params["symbol_sequence"]), size=1))
        imperfect_cps = np.sort(imperfect_cps)
    psts = []
    pid_list = set(run_params["symbol_sequence"])
    cursor = 0
    for cp_idx, cp in enumerate(imperfect_cps):
        start = cursor
        end = cp
        pst = construct_pst(
            pst_id=f"{run_id}:reg_{start}_{end}",
            pid_list=pid_list,
            pid_seq=run_params["symbol_sequence"][start:end],
            subsequence_minimum_occurrence=2,
            subsequence_length_limit=1,
            smoothing_min_p=0.001,
            to_render=False,
            to_save=False,
            output_dir_path=None,
            log_level="DEBUG"
        )
        psts.append(pst)
        cursor = end
    dist_pairs, dist_matrix = match_pst_pairs(
        pst_list=psts,
        matching_function="epstm_2020",
        matching_parameters={"I": 0.5}
    )
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.1,
        metric="precomputed",
        linkage="single"
    )
    clusterer.fit(dist_matrix)
    epstm_icp_labels_pred = []
    for regime_idx, regime_length in enumerate(np.diff(run_params["change_points"], prepend=0)):
        epstm_icp_labels_pred.extend(np.repeat(clusterer.labels_[regime_idx], regime_length))
    epstm_icp_ari = cluster.adjusted_rand_score(labels_pred=epstm_icp_labels_pred, labels_true=labels_true)
    result_dict["epstm_icp:ari"] = epstm_icp_ari

    # CluStream
    X = [[x] for x in run_params["symbol_sequence"]]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_enc = enc.fit_transform(X).toarray()
    labels_pred_clustream = []
    clustream = CluStream(
        n_macro_clusters=params["clustream"]["n_macro_clusters"],
        max_micro_clusters=params["clustream"]["max_micro_clusters"],
        micro_cluster_r_factor=params["clustream"]["micro_cluster_r_factor"],
        time_window=params["clustream"]["time_window"],
        time_gap=params["clustream"]["time_gap"],
        seed=42,
    )
    for i, (x, _) in enumerate(iter_array(X_enc)):
        clustream.learn_one(x)
        if i > 20:
            labels_pred_clustream.append(clustream.predict_one(x))
    length_diff = len(labels_true) - len(labels_pred_clustream)
    clustream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred_clustream, labels_true=labels_true[length_diff:])
    result_dict["clustream:ari"] = clustream_ari

    # DBStream
    X = [[x] for x in run_params["symbol_sequence"]]
    enc = OneHotEncoder(handle_unknown="ignore")
    X_enc = enc.fit_transform(X).toarray()
    labels_pred_dbstream = []
    dbstream = DBSTREAM(
        clustering_threshold=params["dbstream"]["clustering_threshold"],
        fading_factor=params["dbstream"]["fading_factor"],
        cleanup_interval=params["dbstream"]["cleanup_interval"],
        intersection_factor=params["dbstream"]["intersection_factor"],
        minimum_weight=params["dbstream"]["minimum_weight"]
    )
    for i, (x, _) in enumerate(iter_array(X_enc)):
        dbstream.learn_one(x)
        labels_pred_dbstream.append(dbstream.predict_one(x))

    length_diff = len(labels_true) - len(labels_pred_dbstream)
    dbstream_ari = cluster.adjusted_rand_score(labels_pred=labels_pred_dbstream, labels_true=labels_true[length_diff:])
    result_dict["dbstream:ari"] = dbstream_ari

    return result_dict

# run tasks in parallel
test_runs = data["test"]
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        results.append(result)

# format results table
results_df = pd.DataFrame.from_records(results)
results_df_frm = results_df\
    .filter(regex=":ari$")\
    .agg(lambda x: fr"${x.mean():.2f}\pm{x.std():.2f}$")\
    .reset_index()\
    .rename(columns={"index":"alg", 0:"ari"}
)
alg_dict = {
    "emc:ari": "EMC",
    "epstm_tcp:ari": "EPSTM",
    "epstm_icp:ari": "EPSTM",
    "clustream:ari": "CluStream",
    "dbstream:ari": "DBStream",
}
results_df_frm["alg"] = results_df_frm["alg"].replace(alg_dict, regex=True)
results_df_frm.insert(1, "how", ["online","offline","offline","online","online"], True)
results_df_frm.insert(2, "cps", ["not given","given (true)","given (imperfect)","not given","not given"], True)
results_df_frm = results_df_frm.sort_values(by=["how"], ascending=False)

# save results
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)
results_path = os.path.join(results_dir_path, "syn_mdr_ari.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.3f")
print(f"results saved: {results_path}")

# individual runs
results_df_frm_ind = results_df.sort_values(by="emc:ari", ascending=False)[["run_id","emc:ari"]]
results_ind_path = os.path.join(results_dir_path, "syn_mdr_emc_runs.csv")
results_df_frm_ind.to_csv(results_ind_path, index=False, header=True, float_format="%.3f")
print(f"results saved: {results_ind_path}")

# visualize
for run_id in results_df_frm_ind["run_id"].values[:5]:
    # run_id = "run_108"
    run = test_runs[run_id]
    result = results_df.loc[results_df["run_id"] == run_id]
    plot_mode_transition(
        subplots=["true_modes", "discovered_modes"],
        labels_true=np.array(run["subprocess_sequence_full"])+1,
        data=None,
        deviation_history=result["emc:dev_hist"].values.tolist()[0],
        labels_pred=result["emc:labels_pred"].values.tolist()[0],
        change_points=np.insert(run["change_points"], 0, 0),
        cp_scale_coeff=params["emc"]["tau"],
        data_label=None,
        data_yticks=None,
        data_include_modes=False,
        legend_text=f"ARI: {result["emc:ari"].values[0]:.2f}",
        mode_id_to_label_map={mode_id: f"M{mode_id}" for mode_id in np.arange(1, data["meta"]["number_of_subprocesses"]+1)},
        xtick_rotation=45,
        mpl_conf_path=get_mpl_conf_path("pub"),
        # mpl_conf_override={"axes.labelsize": 8},
        fig_size=set_size(width="tpami", aspect_ratio=1.75, fraction= 0.5),
        fig_path=os.path.join(results_dir_path, f"syn_mdr_mt_{run_id}.pdf"),
    )
