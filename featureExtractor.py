import cv2
import numpy as np

np.set_printoptions(suppress=True)

from skimage.measure import ransac      # random sample consensus, deals with outliers
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform        # governs how points correspons to each other, geometric relations of image pairs

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
        return np.dot(self.Kinv, add_ones(pts.T).T)[:, 0:2]     # take only the first 2 columns, dot product of inverse transpose


    def denormalize(self, pt):
        ret = np.dot(self.k, np.array([pt[0], pt[1], 1.0]))         # homogenous matrix dot product with inverse
        # ret /= ret[2]
        return int(round(ret[0])), int(round(ret[1]))


    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)      # strong corners on an image
                                                                                                                            # image needs to be black and white
        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]           # extract keypoints from features
        kps, des = self.orb.compute(img, kps)                                        # compute keypoints and descriptors

        # matching
        ret = []
        if self.last is not None:
            matches = self.bf.knnMatch(des, self.last['des'], k=2)                         # using knn instead of regular match

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
            ret[: ,0 , :] = np.dot(self.Kinv, add_ones(ret[:, 0 ,:]).T).T[:, 0:2]
            ret[: ,1 , :] = np.dot(self.Kinv, add_ones(ret[:, 1 ,:]).T).T[:, 0:2]

            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)

            ret = ret[inliers]       # outlier removal

            # s,v,d = np.linalg.svd(model.params)     # singular value decomposition on matrix
            # print(v)

        self.last = {'kps' : kps, 'des' : des}     # keypoints and descriptors
        return ret

# f is the focal length of the camera 1 radian = how many pixels?
# Fundamental Matrix - uncalibrated camera  coplanarity constraint
# Essential Matrix - calibrated camera

