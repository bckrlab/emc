import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cluster
from tqdm.auto import tqdm

from emc.estimator.EMC import EMC
from emc.utils.data import grouped_decompose
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size

# get paths
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "har", "output")
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)

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

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"{params_path} loaded")
    # print(params)

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
subjects_test = np.setdiff1d(np.arange(1,16), subjects_optim)
number_of_runs = 10
number_of_regimes = 10
master_rng = np.random.default_rng(42)
local_seeds = master_rng.integers(low=0, high=999999, size=number_of_runs, dtype=int)
test_runs = {}
pbar_conf = {"total": len(subjects_test)*number_of_runs}
with tqdm(**pbar_conf) as pbar:
    for run_index in range(number_of_runs):
        rng = np.random.default_rng(local_seeds[run_index])
        # generate a random sequence of activities
        while True:
            rnd_act_idx_seq = rng.choice(a=activities, size=number_of_regimes)
            if 0 not in np.diff(rnd_act_idx_seq):
                break
        for subject in subjects_test:
            # combine data of randomly generated activity sequence
            rnd_act_lbl_seq = [act_idx_lbl_map[a] for a in rnd_act_idx_seq]
            X_seq = pd.DataFrame()
            regime_lengths = []
            for act_lbl in rnd_act_lbl_seq:
                data_path = os.path.join(paths["data"]["realworld_har"]["samples"], f"subject_{subject}", act_lbl, f"{sensor}.csv")
                data = pd.read_csv(data_path, sep=",", header=0, usecols=["attr_x","attr_y","attr_z"])
                X_seq = pd.concat([X_seq, data], ignore_index=True)
                regime_lengths.append(len(data))
        
            X_sc_test = scaler.transform(X_seq)
            grouped_target_signal = grouped_decompose(X_sc_test, prim_len)
            prim_id_seq = alphabet.predict(grouped_target_signal)
            test_runs[f"s{subject}_r{run_index+1}"] = {
                "subprocess_sequence": rnd_act_idx_seq,
                "regime_lengths": regime_lengths,
                "change_points": np.insert(np.cumsum(regime_lengths), 0, 0),
                "discrete_sequence": prim_id_seq
            }
            pbar.update()

def execute_single_run(run_desc):

    run_id, run_params = run_desc
    result_dict = {"run_id": run_id, "subject": run_id.split("_")[0]}

    emc_ins = EMC(
        alpha=alp_car,
        order=1,
        lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
        beta=params["emc"]["beta"],
        delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
        eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
        tau=params["emc"]["tau"],
    )
    emc_ins.process_sequence(run_params["discrete_sequence"], progress=False)

    labels_true = np.repeat(run_params["subprocess_sequence"], run_params["regime_lengths"])
    labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
    length_diff = len(labels_true) - len(labels_pred)
    if length_diff > 0:
        labels_true = labels_true[:-length_diff]
    emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
    result_dict["ari"] = emc_ari

    stationarity_history = np.repeat(emc_ins.stationarity_history, prim_len)
    emc_ari_s = cluster.adjusted_rand_score(
        labels_pred=labels_pred[np.argwhere(stationarity_history)].flatten(),
        labels_true=labels_true[np.argwhere(stationarity_history)].flatten()
    )
    result_dict["ari_s"] = emc_ari_s
    result_dict["drift_ratio"] = np.count_nonzero(stationarity_history==0)/len(stationarity_history)

    return result_dict

# run tasks in parallel
tasks = [(k,p) for k,p in test_runs.items()]
with ProcessPoolExecutor(max_workers=100) as executor:
    futures = [executor.submit(execute_single_run, task) for task in tasks]
    results = []
    for future in tqdm(as_completed(futures), total=len(futures)):
        result = future.result()
        results.append(result)

# format results
results_df = pd.DataFrame.from_records(results)
results_df_frm = results_df\
    .drop(columns=["run_id"])\
    .groupby("subject")\
    .agg(lambda x: fr"${x.mean():.2f}\pm{x.std():.2f}$")\
    .reset_index()\
    .sort_values(by="ari", ascending=False)
results_df_frm["subject"] = results_df_frm["subject"].str.replace("s", "")
avg_row = results_df.copy()\
    .drop(columns=["run_id"])\
    .apply(pd.to_numeric, errors="coerce")\
    .agg(lambda x: fr"${x.mean():.2f}\pm{x.std():.2f}$")\
    .to_frame()\
    .transpose()
avg_row.at[0, "subject"] = "Overall"
results_df_frm = pd.concat([results_df_frm, avg_row], ignore_index=True, sort=False)

# display(results_df_frm)
# display(results_df.sort_values(by="ari", ascending=False)[["run_id","ari"]])

# save results
results_path = os.path.join(results_dir_path, "har_ari.csv")
results_df_frm.to_csv(results_path, index=False, header=True, float_format="%.3f")
print(f"results saved: {results_path}")

# visualization
run = test_runs["s9_r3"]
emc_ins = EMC(
    alpha=alp_car,
    order=1,
    lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
    beta=params["emc"]["beta"],
    delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
    eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
    tau=params["emc"]["tau"],
)
emc_ins.process_sequence(run["discrete_sequence"], progress=True)
labels_true = np.repeat(run["subprocess_sequence"], run["regime_lengths"])
labels_pred = np.repeat(emc_ins.pred_mode_hist, prim_len)
length_diff = len(labels_true) - len(labels_pred)
if length_diff > 0:
    labels_true = labels_true[:-length_diff]
emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)
plot_mode_transition(
    subplots=["true_modes", "discovered_modes"],
    labels_true=labels_true,
    data=None,
    deviation_history=emc_ins.deviation_history,
    labels_pred=labels_pred,
    change_points=run["change_points"],
    cp_scale_coeff=prim_len*emc_ins.tau,
    data_label=None,
    data_yticks=None,
    data_include_modes=False,
    legend_text=f"ARI: {emc_ari:.2f}",
    mode_id_to_label_map={1:"CLIMB↓", 2:"CLIMB↑", 3:"JUMP", 4:"LIE", 5:"RUN", 6:"SIT", 7:"STAND", 8:"WALK"},
    xtick_rotation=45,
    xtick_ids_to_pad=[3,5],
    mpl_conf_path=get_mpl_conf_path("pub"),
    fig_size=set_size(**{"width":"tpami", "aspect_ratio": 1.75, "fraction":0.5}),
    fig_path=os.path.join(results_dir_path, "har_mt.pdf"),
)
