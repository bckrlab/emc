import numpy as np

# compute the Hellinger distance between two probability distributions p and q
def hellinger_distance(p, q):
    return np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2, axis=len(p.shape)-1))
