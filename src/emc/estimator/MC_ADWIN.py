from collections import Counter

import numpy as np
from more_itertools import sliding_window
from river.drift import ADWIN
from tqdm import tqdm


class MC_ADWIN:

	def __init__(
			self,
			alpha,
			order=1,
			delta=0.002,
			clock=32,
			max_buckets=5,
			min_window_length=5,
			grace_period=10

		):
		self.alpha = alpha  # alphabet cardinality
		self.order = order  # markov order

		# initialize ADWIN instance
		self.adwin_ins = ADWIN(
            delta=delta,
            clock=clock,
            max_buckets=max_buckets,
            min_window_length=min_window_length,
			grace_period=grace_period
        )

		# observation history
		self.observed_s = []
		self.observed_i = []

		# symbol <-> index mappings
		self.symbol_index_map = {}
		self.index_symbol_map = {}

		# estimate history
		self.initial_estimate = np.ones(np.repeat(self.alpha, self.order + 1)) / self.alpha
		self.estimates = [self.initial_estimate]
		self.detected_changes = []
		self.window_size_hist = []

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
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

	def build_mc(self, sequence):

		# initialize frequency hypermatrix
		freqs = np.zeros(np.repeat(self.alpha, self.order + 1))

		# get subsequence fragments
		fragments = sliding_window(sequence, self.order+1)

		# count fragments and update freqs
		for fragment, freq in Counter(list(fragments)).items():
			freqs[fragment] = freq

		# calculate probs
		conditional_sum = np.sum(freqs, axis=self.order, keepdims=True)
		probs = np.nan_to_num(np.divide(freqs, conditional_sum))
		probs[np.where(np.sum(freqs, axis=self.order) == 0)] += (1/self.alpha)

		self.estimates.append(probs)

	def process_symbol(self, s):
		self.observed_s.append(s)
		self.observed_i.append(self.get_index(s))

		self.adwin_ins.update(s)
		self.window_size_hist.append(self.adwin_ins.width)

		if self.adwin_ins.drift_detected:
			self.detected_changes.append(len(self.observed_s))
			# print(f"change detected at {len(self.observed_s)}")

		window_i = self.observed_i[-int(self.adwin_ins.width):]
		# print(f"width: {self.adwin_ins.width}, window:{window_i}")

		# if len(self.observation_index_window) >= self.order + 1:
		self.build_mc(window_i)

	# processes each symbol in a given sequence, shows progress information
	def process_sequence(self, sequence, progress=False, progress_desc="", progress_desc_width=8):
		sequence_length = len(sequence)
		if progress:
			for symbol_count in tqdm(
					iterable = range(sequence_length),
					desc = progress_desc.ljust(progress_desc_width),
					bar_format = "{l_bar}{bar:10}{r_bar}{bar:-10b}",
					ascii = True
				):
				self.process_symbol(sequence[symbol_count])
		else:
			for symbol_count in range(sequence_length):
				self.process_symbol(sequence[symbol_count])
