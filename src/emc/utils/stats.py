# efficient calculation of moving statistics

import numpy as np

class MovingStatistics(object):

	def __init__(self):
		self.reset()

	# Welford’s method
	# for unbiased var use:
	# self.var = ((self.var * (self.N - 2)) + (new - new_mean) * (new - old_mean)) / (self.N - 1)
	def add(self, new):
		self.N += 1
		if self.N == 1:
			self.mean = new
			self.var = 0
			self.std = 0
		else:
			old_mean = self.mean
			new_mean = old_mean + (new - old_mean) / self.N
			self.mean = new_mean
			self.var = ((self.var * (self.N - 1)) + (new - new_mean) * (new - old_mean)) / (self.N)
			self.std = np.sqrt(self.var)

	def replace(self, old, new):
		old_mean = self.mean
		new_mean = old_mean + (new - old) / self.N
		self.mean = new_mean
		self.var += (new - old) * (new - new_mean + old - old_mean) / (self.N)
		self.std = np.sqrt(self.var)

	def reset(self):
		self.N = 0
		self.mean = 0
		self.var = 0
		self.std = 0

	def show(self):
		# print("N:{}, mean:{:.4f}, var:{:.4f}, std:{:.4f}".format(self.N, self.mean, self.var, self.std))
		print("ms: mean:{:.4f}, var:{:.4f}, std:{:.4f}".format(self.mean, self.var, self.std))

class ExpectedTensor:

    def __init__(self, initial_tensor):
        self.mean = initial_tensor
        self.count = 0

    def update(self, new_value):
        self.count += 1
        delta = new_value - self.mean
        self.mean += delta / self.count
		