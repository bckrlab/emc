import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from bkmeans import BKMeans
from sklearn.metrics import davies_bouldin_score
from tqdm.autonotebook import tqdm

from emc.utils.data import windowed_decompose
from emc.utils.loader import load_cwru_data
from emc.utils.paths import get_paths

# get paths
paths = get_paths()

# dirs
output_dir_path = os.path.join(paths["exps"], "cwru", "output")
Path(output_dir_path).mkdir(parents=True, exist_ok=True)
alphabet_dir_path = os.path.join(output_dir_path, "alphabet")
Path(alphabet_dir_path).mkdir(parents=True, exist_ok=True)

# load data
X_alp, cp_alp, y_true_alp = load_cwru_data(condition_sequence=["L0-IR:07","L0-IR:28"], points_per_condition=12000)

# decomposition
prim_len = 2
X_alp_dec = windowed_decompose(X_alp, prim_len)

# clustering
clue = []
n_clusters_list = np.arange(6,41)
for n_clusters in tqdm(n_clusters_list):
    clusterer = BKMeans(n_clusters=n_clusters, m=5, random_state=42)
    X_alp_cls = clusterer.fit_predict(X_alp_dec)
    db = davies_bouldin_score(X_alp_dec, X_alp_cls)
    clue.append({"n_clusters": n_clusters, "davies_bouldin_score": db, "inertia": clusterer.inertia_})
clue_df = pd.DataFrame.from_records(clue)

# plot db scores
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    db_plot = sns.lineplot(data=clue_df, x="n_clusters", y="davies_bouldin_score", ax=ax)
    ax.set_xticks(n_clusters_list)
    db_plot.get_figure().savefig(os.path.join(alphabet_dir_path, f"pl:{prim_len}_db.pdf"), bbox_inches="tight")
    plt.close()

# save alphabet
alp_car = 27
clusterer = BKMeans(n_clusters=alp_car, m=30, random_state=42)
clusterer.fit(X_alp_dec)
alp_id = f"ac:{alp_car}_pl:{prim_len}"
with open(os.path.join(alphabet_dir_path, f"{alp_id}.pkl"), "wb") as file:
    pickle.dump(clusterer, file)

# plot primitives
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,4))
    start = 0
    ticks = [0]
    for prm_ind, prm in enumerate(clusterer.cluster_centers_):
        end = start + prim_len
        ax.plot(np.arange(start, end), prm)
        start = end
        ticks.append(start)
    ax.set_xticks(ticks)
    fig.savefig(os.path.join(alphabet_dir_path, f"{alp_id}_primitives.pdf"), bbox_inches="tight")
    plt.close()

# plot cluster centers
with sns.axes_style("whitegrid"):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,6))
    sns.scatterplot(x=X_alp_dec[:, 0], y=X_alp_dec[:, 1], s=50, alpha=0.6, edgecolor='k', ax=ax)
    sns.scatterplot(x=clusterer.cluster_centers_[:, 0], y=clusterer.cluster_centers_[:, 1], ax=ax)
    fig.savefig(os.path.join(alphabet_dir_path, f"{alp_id}_primitives_cc.pdf"), bbox_inches="tight")
    plt.close()
