import os
import pickle
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from emc.generator.markov import MarkovianSwitchingSystem
from emc.utils.paths import get_paths

# synthetic data parameters
alphabet_cardinality = 4
mode_process_order = 1
subprocess_order = 1
number_of_subprocesses = 5
number_of_regimes = 10
regime_length_bounds_list = [[500,1000],[1500,2000],[2500,3000]]
number_of_runs = [10,100]

# generate data
optm_runs = {}
test_runs = {}
for regime_length_bounds in regime_length_bounds_list:
    desc = f"{regime_length_bounds[0]}:{regime_length_bounds[1]}".ljust(10)
    for run_count in tqdm(range(np.sum(number_of_runs)), desc=desc):
        mss_ins = MarkovianSwitchingSystem(
            alphabet_cardinality=alphabet_cardinality,
            mode_process_order=mode_process_order,
            subprocess_order=subprocess_order,
            number_of_subprocesses=number_of_subprocesses,
            rng_or_seed=run_count
        )
        mss_ins.generate_subprocess_sequence(
            number_of_regimes=number_of_regimes,
            avoid_loops=True
        )
        symbol_sequence = mss_ins.generate_symbol_sequence(
            regime_length_bounds=regime_length_bounds
        )
        run_params = {
            "subprocess_sequence": mss_ins.subprocess_sequence,
            "regime_lengths": mss_ins.regime_lengths,
            "change_points": np.cumsum(mss_ins.regime_lengths),
            "subprocesses": mss_ins.subprocesses,
            "subprocess_sequence_full": np.repeat(mss_ins.subprocess_sequence, mss_ins.regime_lengths),
            "symbol_sequence": symbol_sequence
        }
        if run_count < number_of_runs[0]:
            optm_runs[f"reg{':'.join(map(str, regime_length_bounds))}_run{run_count}"] = run_params
        else:
            test_runs[f"reg{':'.join(map(str, regime_length_bounds))}_run{run_count}"] = run_params

# prepare data
data = {
    "meta": {
        "alp_car": alphabet_cardinality,
        "k": mode_process_order,
        "number_of_subprocesses": number_of_subprocesses,
        "number_of_regimes": number_of_regimes,
        "regime_length_bounds_list": regime_length_bounds_list,
    },
    "optm": optm_runs,
    "test": test_runs
}

# save data
paths = get_paths()
output_dir_path = os.path.join(paths["exps"], "synthetic", "01_pt", "output")
Path(output_dir_path).mkdir(parents=True, exist_ok=True)
data_path = os.path.join(output_dir_path, "data.pkl")
with open(data_path, "wb") as file:
    pickle.dump(data, file)
print(f"data saved: {data_path}")
