import numpy as np

from emc.utils.compare import hellinger_distance

# this class implements a generic Markov chain
class MarkovChain:

	def __init__(self, alphabet_cardinality, order, rng_or_seed):
		self.alphabet_cardinality = alphabet_cardinality
		self.alphabet = [p for p in range(self.alphabet_cardinality)]
		self.order = order
		self.rng = np.random.default_rng(rng_or_seed) if isinstance(rng_or_seed, int) else rng_or_seed
		self.transition_matrix = None

	# generates a single transition matrix with specified method
	def generate_transition_matrix(self, method=None, params=None):

		# each CPD is a flat dirichlet distribution
		if method is None or method == "flat_dirichlet":
			transition_matrix = self.rng.dirichlet(
				alpha=np.repeat(1, self.alphabet_cardinality),
				size=np.repeat(self.alphabet_cardinality, self.order)
			)

		# each CPD is a dirichlet distribution with given alpha values
		elif method == "dirichlet":
			transition_matrix = self.rng.dirichlet(
				alpha=params["alpha"],
				size=np.repeat(self.alphabet_cardinality, self.order)
			)

		else:
			return None

		self.transition_matrix = transition_matrix

	# generates a sequence of a given length from the Markov chain
	def generate_sequence(self, sequence_length):
		sequence = self.rng.choice(a=self.alphabet, size=self.order).tolist()
		for i in range(sequence_length - self.order):
			cpd_index = tuple(sequence[-self.order:])
			next_state = self.rng.choice(a=self.alphabet, p=self.transition_matrix[cpd_index])
			sequence.extend([next_state])
		return sequence

# this class implements a generic Markovian switching system (or environment)
class MarkovianSwitchingSystem:

	def __init__(
		self,
		alphabet_cardinality,
		mode_process_order,
		subprocess_order,
		number_of_subprocesses,
		rng_or_seed
	):
		self.alphabet_cardinality = alphabet_cardinality
		self.mode_process_order = mode_process_order
		self.subprocess_order = subprocess_order
		self.number_of_subprocesses = number_of_subprocesses
		self.rng = np.random.default_rng(rng_or_seed) if isinstance(rng_or_seed, int) else rng_or_seed
		self.subprocesses = {}
		self.subprocess_sequence = []
		self.regime_lengths = []

		# generate the mode process
		self.mode_process = MarkovChain(
			alphabet_cardinality=number_of_subprocesses,
			order=self.mode_process_order,
			rng_or_seed=self.rng
		)
		self.mode_process.generate_transition_matrix()

		# generate subprocesses
		self.generate_subprocesses_with_min_dist(min_dist_threshold=0.2)

	# generates subprocesses (each is a separate Markov chain) so that the minimum distance of any pair is more than
	# a given threshold
	def generate_subprocesses_with_min_dist(self, min_dist_threshold, agg_func="max"):
		for subprocess_count in range(self.number_of_subprocesses):
			while True:
				subprocess_ = MarkovChain(
					alphabet_cardinality=self.alphabet_cardinality,
					order=self.subprocess_order,
					rng_or_seed=self.rng
				)
				subprocess_.generate_transition_matrix()
				min_dist = 1
				for existing_subprocess_id, existing_subprocess in self.subprocesses.items():
					if agg_func == "max":
						dist = np.max(hellinger_distance(
							existing_subprocess.transition_matrix,
							subprocess_.transition_matrix
						))
					elif agg_func == "mean":
						dist = np.mean(hellinger_distance(
							existing_subprocess.transition_matrix,
							subprocess_.transition_matrix
						))
					else:
						print("unknown agg. func.")
						return
					if dist < min_dist:
						min_dist = dist
				if min_dist > min_dist_threshold:
					break
			self.subprocesses[subprocess_count] = subprocess_

	# generates subprocesses (each is a separate Markov chain) randomly
	def generate_subprocesses(self):
		for subprocess_count in range(self.number_of_subprocesses):
			subprocess_ = MarkovChain(
				alphabet_cardinality=self.alphabet_cardinality,
				order=self.subprocess_order,
				rng_or_seed=self.rng
			)
			subprocess_.generate_transition_matrix()
			self.subprocesses[subprocess_count] = subprocess_

	# generates a random subprocess sequence for the MSS
	def generate_subprocess_sequence(self, number_of_regimes, avoid_loops=True):
		subprocess_sequence = self.mode_process.generate_sequence(number_of_regimes)
		if avoid_loops:
			while np.count_nonzero(np.diff(subprocess_sequence)==0):
				subprocess_sequence = self.mode_process.generate_sequence(number_of_regimes)
		self.subprocess_sequence = subprocess_sequence

	# generates random symbol sequences from each subprocess
	# regime lengths are randomly selected from a uniform distribution described by lower and upper bounds
	def generate_symbol_sequence(self, regime_length_bounds):
		self.regime_lengths = self.rng.integers(
			low=regime_length_bounds[0],
			high=regime_length_bounds[1],
			size=len(self.subprocess_sequence)
		)
		symbol_sequence = []
		for subprocess_count, subprocess_id in enumerate(self.subprocess_sequence):
			regime_symbol_sequence = self.subprocesses[subprocess_id].generate_sequence(
				sequence_length=self.regime_lengths[subprocess_count]
			)
			symbol_sequence.extend(regime_symbol_sequence)
		return symbol_sequence
