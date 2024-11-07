from itertools import product

import numpy as np


# aligns a given hypermatrix based on the provided index-symbol mapping
def align_hypermatrix(
		hypermatrix,  # hypermatrix to be aligned, assumed to have P("1"|"0") at (0, 1)
		index_symbol_map
	):
	order = len(hypermatrix.shape) - 1 # markov order of the hypermatrix
	hypermatrix = np.array(hypermatrix)
	aligned_hypermatrix = np.empty(hypermatrix.shape)

	indices = list(product(index_symbol_map.keys(), repeat=order+1))
	for index in indices:
		sim_index = tuple([int(index_symbol_map[i]) for i in index])
		aligned_hypermatrix[index] = hypermatrix[sim_index]
	return aligned_hypermatrix

# evaluates the estimates against true matrices by calculating the mean absolute error (MAE) and absolute errors
def evaluate_estimates(estimates, true_matrices, regime_lengths, index_symbol_map):

	true_matrices = np.array(true_matrices)
	order = len(true_matrices[0].shape) - 1
	number_of_regimes = len(true_matrices)

	# check regime lengths
	if isinstance(regime_lengths, list):
		if len(regime_lengths) != number_of_regimes:
			print("Missing regime length.")
			return None
		regime_lengths = np.array(regime_lengths)
	elif isinstance(regime_lengths, np.ndarray):
		pass
	else:
		regime_lengths = np.repeat(regime_lengths, number_of_regimes)

	# calculate absolute errors
	absolute_errors = []
	for true_matrix_index, true_matrix in enumerate(true_matrices):
		start = np.sum(regime_lengths[:true_matrix_index])
		end = start + regime_lengths[true_matrix_index]
		if index_symbol_map:
			aligned_true_matrix = align_hypermatrix(true_matrix, index_symbol_map)
			regime_absolute_errors = np.mean(
				np.abs(estimates[start:end] - aligned_true_matrix),
				axis=tuple(range(1, order+2))
			)
		else:
			regime_absolute_errors = np.sum(
				np.abs(estimates[start:end] - true_matrix),
				axis=tuple(range(1, order+2))
			)
		absolute_errors.extend(regime_absolute_errors)
	
	# calculate mean absolute error
	total_absolute_error = np.sum(absolute_errors)
	mae = total_absolute_error / np.sum(regime_lengths)

	return mae, absolute_errors

# evaluates the performance of change point detection
def evaluate_cpd(true_cps, detected_cps, margin_of_error, allow_prior, sequence_length):

	true_cps = np.array(true_cps)
	detected_cps = np.array(detected_cps)

	eval_log = {"tp": [], "fp": [], "fn": []}
	found_true_cps = []
	tp_lags = []
	for detected_cp in detected_cps:

		closest_tcp = true_cps[np.argmin(np.abs(detected_cp - true_cps))]
		lag = detected_cp - closest_tcp
		if allow_prior:
			lag = np.abs(lag)

		if closest_tcp in found_true_cps:
			if lag > margin_of_error:
				eval_log["fp"].append(detected_cp)
			continue

		if 0 < lag < margin_of_error:
			eval_log["tp"].append(detected_cp)
			found_true_cps.append(closest_tcp)
			tp_lags.append(lag)
		else:
			eval_log["fp"].append(detected_cp)

	for miss in np.setdiff1d(true_cps, found_true_cps):
		eval_log["fn"].append(miss)

	# metrics
	tp = len(eval_log["tp"])
	fp = len(eval_log["fp"])
	tn = sequence_length - len(true_cps) - fp
	fn = len(eval_log["fn"])
	fpr = fp/(fp+tn)
	fnr = fn/(fn+tp)
	f1 = (2*tp)/(2*tp+fp+fn)
	mdl = np.mean(tp_lags) if len(tp_lags) > 0 else np.nan
	return {
		"eval_log": eval_log,
		"tp": tp,
		"fp": fp,
		"tn": tn,
		"fn": fn,
		"fpr": fpr,
		"fnr": fnr,
		"f1": f1,
		"mdl": mdl
	}
