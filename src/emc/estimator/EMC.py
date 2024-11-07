import logging
from collections import deque

import numpy as np
from tqdm import tqdm

from emc.utils.compare import hellinger_distance
from emc.utils.stats import ExpectedTensor


class EMC:

	def __init__(
		self,
		alpha, # int, alphabet cardinality, it is needed to initialize the estimate tensor
		order, # int, Markov order of the chain
		lambda_, # list or double [0,1], AKA: learning coefficient
		beta, # double [0,1], entropy regularization rate
		delta, # list or double [0,1], min distance to fire drift detection
		eta, # list or double [0,1], model similarity threshold
		tau, # int, distance between compared estimates in terms of symbols
		live_memory=True, # boolean, whether to update models in memory
		symbols_are_indices=True, # boolean, whether to use symbols as indices in estimate tensor
		log_file_path=None # path, file to write logs
	):

		# initialize EMC instance
		self.alpha = alpha
		self.order = order
		self.lambda_fast, self.lambda_slow = lambda_[:2] if isinstance(lambda_, list) else (lambda_, lambda_)
		self.beta = beta
		self.delta_fast, self.delta_slow = delta[:2] if isinstance(delta, list) else (delta, delta)
		self.eta_fast, self.eta_slow = eta[:2] if isinstance(eta, list) else (eta, eta)
		self.tau = tau
		self.live_memory = live_memory
		self.symbols_are_indices = symbols_are_indices
		self.is_stationary = False # start from non-stationary state
		self.dist = hellinger_distance
		self.obs_count = 0

		# estimate and prediction histories
		self.P0 = 1/self.alpha * np.ones(shape=np.repeat(self.alpha, self.order+1))
		self.P_hist = [self.P0]
		self.P_exp_hist = []
		self.curr_mode_pred = None # id of the most recently predicted mode
		self.pred_mode_hist = [] # predicted mode history

		# symbol to index and index to symbol maps
		self.symbol_index_map = {}
		self.index_symbol_map = {}

		# condition vectors (symbol and index)
		self.cnd_vec = deque(maxlen=self.order)
		self.cndi_vec = deque(maxlen=self.order)

		# detections
		self.detected_drifts = []
		self.detected_changes = []
		self.deviation_history = []
		self.stationarity_history = []

		# the memory
		self.learned_modes = {}

		# initialize logger
		if log_file_path:
			logging.basicConfig(
				filename=log_file_path,
				filemode="w",
				format="%(levelname)s: %(message)s",
				encoding="utf-8",
				level=logging.INFO,
				force=True
			)

	# EMC update (Eq. 2)
	# - updates the probabilities of the observed and non-observed relevant events
	# - probabilities of the irrelevant events are not updated
	def update_estimate(self, symbol_index):
		lambda_ = self.lambda_slow if self.is_stationary else self.lambda_fast
		new_estimate = np.copy(self.P_hist[-1])
		new_estimate[tuple(self.cndi_vec)] *= lambda_
		new_estimate[tuple(self.cndi_vec)][symbol_index] += (1 - lambda_)
		self.P_hist.append(new_estimate)

	# entropy regularization (Eq. 11)
	# - gradually makes the probability tensor closer to uniform distribution
	# - only affects the irrelevant region
	def entropy_regularization(self):
		uniform_tensor = np.copy(self.P0)
		exclude_idx = tuple(self.cndi_vec)
		uniform_tensor[exclude_idx] = self.P_hist[-1][exclude_idx]
		self.P_hist[-1] = (1 - self.beta) * self.P_hist[-1] + self.beta * uniform_tensor
	
	# compares the most recent estimate with a past version, changes steady/drift (stationary/non-stationary) state
	def check_deviation(self):

		# calculate the deviation
		current_deviation = np.max(self.dist(self.P_hist[-1], self.P_hist[-self.tau]))
		self.deviation_history.append(current_deviation)

		if self.is_stationary:

			# S -> NS
			if current_deviation > self.delta_slow: 
				logging.info(f"{self.obs_count}: stationary -> non-stationary (H:{current_deviation:.4f}, Ds:{self.delta_slow})")
				self.detected_drifts.append(self.obs_count)
				self.is_stationary = False
				self.identify_regime()

			# S -> S
			else:
				logging.info(f"{self.obs_count}: stationary (H:{current_deviation:.4f}, Ds:{self.delta_slow})")

		else:

			# NS -> NS
			if current_deviation > self.delta_fast:
				logging.info(f"{self.obs_count}: non-stationary (H:{current_deviation:.4f}, Df:{self.delta_fast})")
				self.identify_regime()

			# NS -> S
			else:
				logging.info(f"{self.obs_count}: non-stationary -> stationary (H:{current_deviation:.4f}), checking memory")
				self.is_stationary = True
				self.identify_regime()

	# compares the current estimate to saved models in the memory
	# updates the memory if needed
	def identify_regime(self):

		# get a copy of the most recent estimate
		current_matrix = np.copy(self.P_hist[-1])

		# memory is empty
		if len(self.learned_modes) == 0:

			# save the mode
			if self.is_stationary:
				self.insert_to_memory(current_matrix)

		# memory is NOT empty
		else:

			mode_dists = self.get_dists_from_memory(self.P_hist[-1])
			closest_model_id, min_distance = list(mode_dists.items())[0]
			logging.info(f"dists: {self.get_dists_from_memory(self.P_hist[-1])}")

			# closest model is close enough
			eta = self.eta_slow if self.is_stationary else self.eta_fast
			if min_distance < eta:
				self.curr_mode_pred = closest_model_id
				logging.info(f"previously seen mode {closest_model_id} is close enough (t:{eta})")

			# closest model is NOT close enough
			else:
				if self.is_stationary:
					logging.info("no mode is similar enough (t: {})".format(eta))
					self.insert_to_memory(current_matrix)

	# inserts a given tensor to the memory, allocates the next available index
	def insert_to_memory(self, tensor_to_insert):
		new_mode_id = len(self.learned_modes)+1
		self.learned_modes[new_mode_id] = ExpectedTensor(initial_tensor=tensor_to_insert)
		self.curr_mode_pred = new_mode_id
		logging.info("{}: new mode {} is saved".format(self.obs_count, new_mode_id))

	# calculates the distance of a given tensor to the elements of the memory
	# returns a sorted dict of distances
	def get_dists_from_memory(self, ref_tensor):
		mode_dists = {}
		for mode_id, mode_transition_matrix in self.learned_modes.items():
			mode_dists[mode_id] = np.max(self.dist(mode_transition_matrix.mean, ref_tensor))
		return dict(sorted(mode_dists.items(), key=lambda item: item[1]))

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
		if self.symbols_are_indices:
			if symbol not in self.symbol_index_map:
				self.symbol_index_map[symbol] = symbol
				self.index_symbol_map[symbol] = symbol
			index = symbol
		else:
			if len(self.symbol_index_map) == 0:
				self.symbol_index_map[symbol] = 0
				self.index_symbol_map[0] = symbol
			elif symbol not in self.symbol_index_map:
				if len(self.symbol_index_map) >= self.alpha:
					print(f"Unexpected symbol observed: {symbol}")
					exit()
				next_available_index = max(self.symbol_index_map.values()) + 1
				self.symbol_index_map[symbol] = next_available_index
				self.index_symbol_map[next_available_index] = symbol
			index = self.symbol_index_map[symbol]
		return index

	# processes a single symbol
	def process_symbol(self, symbol):

		# learn
		index = self.get_index(symbol)
		if len(self.cndi_vec) == self.order:
			self.update_estimate(index)
			if self.beta > 0:
				self.entropy_regularization()
		else:
			self.P_hist.append(self.P0)

		# maintain memory
		if len(self.learned_modes) == 0:
			self.pred_mode_hist.append(1)
			P_exp = np.copy(self.P_hist[-1])
		else:
			if self.curr_mode_pred != self.pred_mode_hist[-1]:
				self.detected_changes.append(self.obs_count)
			self.pred_mode_hist.append(self.curr_mode_pred)
			if self.is_stationary:
				if self.live_memory:
					self.learned_modes[self.curr_mode_pred].update(self.P_hist[-1])
				P_exp = np.copy(self.learned_modes[self.curr_mode_pred].mean)
			else:
				P_exp = np.copy(self.P_hist[-1])
		self.P_exp_hist.append(P_exp)

		self.stationarity_history.append([0,1][self.is_stationary])
		self.obs_count += 1
		self.cnd_vec.append(symbol)
		self.cndi_vec.append(index)
		if self.obs_count % self.tau == 0:
			self.check_deviation()

	# processes each symbol in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		if progress:
			pbar_conf = {
				"total": len(sequence),
				"desc": progress_desc.ljust(progress_desc_width),
				"bar_format": "{l_bar}{bar:20}{r_bar}{bar:-20b}",
				"ascii": True
			}
			with tqdm(**pbar_conf) as pbar:
				for symbol in sequence:
					self.process_symbol(symbol)
					pbar.update()
		else:
			for symbol in sequence:
				self.process_symbol(symbol)
