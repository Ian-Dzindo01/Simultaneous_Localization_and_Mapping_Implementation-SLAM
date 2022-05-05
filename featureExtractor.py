import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform, EssentialMatrixTransform        # governs how points correspons to each other

class FeatureExtractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self, w, h):
        self.orb = cv2.ORB_create(100)   # 1000 nfeatures
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)        # brute force matcher, norm type
        self.last = None
        self.w, self.h = w,h

    def denormalize(pt):
        return int(round(pt[0] + self.w)), int(round(pt[1] + self.h))


    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)    # strong corners on an image
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


            # noormalize coordinates: subtract to move to 0
            ret[:, :, 0] -= img.shape[0]//2
            ret[:, :, 1] -= img.shape[1]//2

            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=1,
                                    max_trials=100)

            ret = ret[inliers]       # inlier removal

            s,v,d = np.linalg.svd(model.params)     # singular value decomposition on matrix
            print(v)

        self.last = {'kps' : kps, 'des' : des}     # keypoints and descriptors
        return ret

# f is the focal length of the camera 1 radian = how many pixels?
