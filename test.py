import numpy as np

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


s = np.array([1,2], [1,2])
s1 = add_ones(s)
print(s1)
