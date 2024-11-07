import os
from pathlib import Path

import numpy as np
from sklearn.metrics import cluster

from emc.estimator.EMC import EMC
from emc.utils.loader import load_json_file, load_pickle_file
from emc.utils.paths import get_paths
from emc.utils.plot import get_mpl_conf_path, plot_mode_transition, set_size

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "eeg", "output")
results_dir_path = os.path.join(output_dir_path, "results")
Path(results_dir_path).mkdir(parents=True, exist_ok=True)

# load data
data_path = os.path.join(output_dir_path, "data.pkl")
data = load_pickle_file(data_path)
if data is not None:
    print(f"{data_path} loaded")

# load parameters
params_path = os.path.join(output_dir_path, "params.json")
params = load_json_file(params_path)
if params is not None:
    print(f"\n{params_path} loaded.")
    # print(params)

# execute test run
opt_index = 3000
emc_ins = EMC(
    alpha=data["meta"]["n_clusters"],
    order=params["emc"]["order"],
    lambda_=[params["emc"]["lambda_f"], params["emc"]["lambda_s"]],
    beta=params["emc"]["beta"],
    delta=[params["emc"]["delta_f"], params["emc"]["delta_s"]],
    eta=[params["emc"]["eta_f"], params["emc"]["eta_s"]],
    tau=params["emc"]["tau"],
)
emc_ins.process_sequence(data["ms_seq"], progress=False)

# ARI
labels_true = data["Y"][opt_index:]
labels_pred = np.array(emc_ins.pred_mode_hist)[opt_index:]
emc_ari = cluster.adjusted_rand_score(labels_pred=labels_pred, labels_true=labels_true)

stationarity_history = emc_ins.stationarity_history[opt_index:]
emc_ari_s = cluster.adjusted_rand_score(
    labels_pred=labels_pred[np.argwhere(stationarity_history)].flatten(),
    labels_true=labels_true[np.argwhere(stationarity_history)].flatten()
)

print(f"ARI  : {emc_ari:.2f}")
print(f"ARI-S: {emc_ari_s:.2f}")

mode_id_to_label_map = {1: "eyes\nopen", 2: "eyes\nclosed"}
change_points = np.where(np.diff(data["Y"], prepend=np.nan))[0].tolist()
mpl_conf_override = {"axes.labelsize":6, "font.size":6, "xtick.labelsize":4, "ytick.labelsize":6, "legend.fontsize":6}
plot_mode_transition(
    subplots=["true_modes","data","discovered_modes"],
    labels_true=data["Y"]+1,
    data=data["ms_seq"],
    deviation_history=emc_ins.deviation_history,
    labels_pred=emc_ins.pred_mode_hist,
    change_points=change_points,
    cp_scale_coeff=emc_ins.tau,
    data_label="Microstate\nSequence",
	data_yticks=[-1,0,1,2,3,4,5,6,7,8],
    data_include_modes=False,
    # ns_regions=ns_regions,
    legend_text=f"ARI: {emc_ari:.2}",
    legend_loc="best",
    mode_id_to_label_map=mode_id_to_label_map,
    xtick_rotation=60,
    xtick_ids_to_pad=[0,18],
    xtick_ids_to_skip=[7,16,18,19],
    mpl_conf_path=get_mpl_conf_path("pub"),
    mpl_conf_override=mpl_conf_override,
    fig_size=set_size(**{"width":"tpami", "aspect_ratio": 1.25, "fraction":0.5}),
    fig_path=os.path.join(results_dir_path, "eeg_mt.pdf"),
)
print(f"figure saved: {results_dir_path}/eeg_mt.pdf")
