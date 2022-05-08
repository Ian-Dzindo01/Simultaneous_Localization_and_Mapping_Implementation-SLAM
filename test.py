import numpy as np

W = 1920//2
H = 1080//2

K = np.array(([1,0,W//2], [0,1,H//2], [0,0,1]))    # intrinsic matrix
Kinv = np.linalg.inv(K)

def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)       # x.shape[0] and 1 are a single parameter

def normalize(pts):
    return np.dot(Kinv, add_ones(pts.T).T)[:, 0:2]    # dot product of inverse and

s = np.array([[1,2,3,4], [5,6,7,8]])
s1 = add_ones(s)
print(K)
print(Kinv)
