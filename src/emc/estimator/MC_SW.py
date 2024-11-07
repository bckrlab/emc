from collections import Counter, deque

import numpy as np
from more_itertools import sliding_window
from tqdm import tqdm

np.seterr(divide="ignore", invalid="ignore")

class MC_SW:

	def __init__(self, order, alpha, window_size):

		self.order = order
		self.alpha = alpha  # int, alphabet cardinality
		self.window_size = window_size

		# estimate history
		self.initial_estimate = np.ones(np.repeat(self.alpha, self.order + 1)) / self.alpha
		self.estimates = [self.initial_estimate]

		# predicted symbol history
		self.predicted_symbols = []

		# symbol to index and index to symbol maps
		self.symbol_index_map = {}
		self.index_symbol_map = {}

		self.observation_symbol_window = deque(maxlen=self.window_size)
		self.observation_index_window = deque(maxlen=self.window_size)

	def calculate_p(self):

		# initialize frequency hypermatrix
		freqs = np.zeros(np.repeat(self.alpha, self.order + 1))

		# get subsequence fragments
		fragments = sliding_window(self.observation_index_window, self.order+1)

		# count fragments and update freqs
		for fragment, freq in Counter(list(fragments)).items():
			freqs[fragment] = freq

		# calculate probs
		conditional_sum = np.sum(freqs, axis=self.order, keepdims=True)
		probs = np.nan_to_num(np.divide(freqs, conditional_sum))
		probs[np.where(np.sum(freqs, axis=self.order) == 0)] += (1/self.alpha)

		self.estimates.append(probs)

	# returns the predicted next symbol
	def predict_next_symbol(self):
		relevant_cpd = np.copy(self.estimates[-1][tuple(self.observation_index_window)[-self.order:]])
		return self.index_symbol_map[np.argmax(relevant_cpd)]

	# manages symbol->index and index->symbol mappings
	# - assigns a new index to given symbol if it has not been observed before
	# - returns index
	def get_index(self, symbol):
		if len(self.symbol_index_map) == 0:
			self.symbol_index_map[symbol] = 0
			self.index_symbol_map[0] = symbol
		elif symbol not in self.symbol_index_map:
			if len(self.symbol_index_map) >= self.alpha:
				print("Unexpected symbol observed: {}".format(symbol))
				exit()
			next_available_index = max(self.symbol_index_map.values()) + 1
			self.symbol_index_map[symbol] = next_available_index
			self.index_symbol_map[next_available_index] = symbol
		index = self.symbol_index_map[symbol]
		return index

	# processes a single symbol
	def process_symbol(self, symbol):
		index = self.get_index(symbol)
		self.observation_symbol_window.append(symbol)
		self.observation_index_window.append(index)
		if len(self.observation_index_window) >= self.order + 1:
			self.calculate_p()
		self.predicted_symbols.append(self.predict_next_symbol())

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
