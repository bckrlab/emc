import numpy as np
from more_itertools import grouper, windowed

# decomposes the data into fragments of a specified length using a sliding window approach
# each fragment is flattened and concatenated
def windowed_decompose(data, fragment_length):
    data = np.asarray(data)
    number_of_samples, number_of_features = data.shape
    decomposed_features = [list(windowed(data[:, i], fragment_length)) for i in range(number_of_features)]
    decomposed_data = np.array([np.hstack(window) for window in zip(*decomposed_features)])
    return decomposed_data

# groups the data into fragments of a specified length, discarding incomplete groups
# each fragment is flattened and concatenated
def grouped_decompose(data, fragment_length):
    data = np.asarray(data)
    number_of_samples, number_of_features = data.shape
    grouped_features = [
        list(grouper(data[:, i], fragment_length, incomplete="ignore"))
        for i in range(number_of_features)
    ]
    grouped_data = np.array([np.hstack(group) for group in zip(*grouped_features)])
    return grouped_data
