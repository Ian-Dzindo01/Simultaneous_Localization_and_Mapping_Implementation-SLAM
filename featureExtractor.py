import cv2
import numpy as np
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform       # governs how points correspons to each other

class FeatureExtractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(100)   # 1000 nfeatures
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)        # brute force matcher, norm type
        self.last = None

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

        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)

            # outlier inlier filtering
            model, inliers = ransac((ret[:,0], ret[:,1]),
                                    FundamentalMatrixTransform,
                                    min_samples=8,
                                    residual_threshold=0.01,
                                    max_trials=100)



            print(sum(inliers))


        self.last = {'kps' : kps, 'des' : des}     # keypoints and descriptors
        return ret


# train image is the one learned, query is the image you are trying to match with the
# ones already trained
