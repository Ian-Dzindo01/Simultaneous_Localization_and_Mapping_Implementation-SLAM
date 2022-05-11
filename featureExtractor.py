import cv2
import numpy as np

np.set_printoptions(suppress=True)

from skimage.measure import ransac      # random sample consensus, deals with outliers
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform        # governs how points correspons to each other, geometric relations of image pairs

f_est_avg = []

# [[x, y]] -> [[x,y,1]]
def add_ones(x):
    return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)   # pad with ones (parameter 1)

class FeatureExtractor(object):
    GX = 16//2        # ??
    GY = 12//2        # ??

    def __init__(self, K):
        self.orb = cv2.ORB_create(100)                   # 1000 nfeatures
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)        # brute force matcher, norm type, using create(), cause previous is obsolete, NORM_HAMMING best for ORB
        self.last = None
        self.k = K
        self.Kinv = np.linalg.inv(self.k)            # matrix inverse


    def normalize(self, pts):
        return np.dot(self.Kinv, add_ones(pts).T).T[:, 0:2]     # take only the first 2 columns, dot product of inverse transpose


    def denormalize(self, pt):
        ret = np.dot(self.k, np.array([pt[0], pt[1], 1.0]))         # homogenous matrix dot product with inverse
        # maybe?
        # ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))


    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)      # strong corners on an image
                                                                                                                            # image needs to be black and white
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]           # extract keypoints from features
        kps, des = self.orb.compute(img, kps)

        # matching
        ret = []
        if self.last is not None:        # last not empty
            matches = self.bf.knnMatch(des, self.last['des'], k=2)                         # using knn instead of regular match, 2 best matchess, matches current with previous frame

            # ratio test
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1, kp2))


        # outlier inlier filtering using fundamental matrices
        if len(ret) > 0:
            ret = np.array(ret)


            # normalize coordinates: subtract to move to 0
            ret[: ,0 , :] = self.normalize(ret[:, 0 , :])
            ret[: ,1 , :] = self.normalize(ret[: ,1 , :])

            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    # FundamentalMatrixTransform,
                                    EssentialMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=0.001,          # for him it is 0.005
                                    max_trials=100)

            ret = ret[inliers]       # outlier removal

            u,w,vt = np.linalg.svd(model.params)     # singular value decomposition on matrix
            W = np.mat([[0,-1,0],[1,0,0],[0,0,1]], dtype=float)

            assert np.linalg.det(u) > 0

            if np.linalg.det(vt) < 0:
                 vt *= -1.0

            R1 = np.dot(np.dot(u, W), vt)
            R2 = np.dot(np.dot(u, W.T), vt)

            print(R1)
            print(R2)

        self.last = {'kps' : kps, 'des' : des}     # keypoints and descriptors
        return ret

# f is the focal length of the camera 1 radian = how many pixels?
# Fundamental Matrix - uncalibrated camera  coplanarity constraint - both used for describing geometric relations between pairs of images
# Essential Matrix - calibrated camera
# We have to get focal length of center - distance between lens and sensor in camera

# f_est = np.sqrt(2)/((vt[0] + vt[1])/2)
# print(f_est, np.median(f_est_avg))
# f_est_avg.append(f_est)

# v should be sqrt(2), sqrt(2) and 0?
# we can use svd on essential matrix to estimate translation and rotation
