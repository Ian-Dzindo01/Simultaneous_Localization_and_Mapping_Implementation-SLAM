import cv2
import numpy as np


class FeatureExtractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(100)   # 1000 nfeatures
        self.bf = cv2.BFMatcher()        # brute force matcher
        self.last = None

    def extract(self, img):
        # detection
        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)    # strong corners on an image

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        # matching
        matches = None
        if self.last is not None:
            matches = self.bf.match(des, self.last['des'])
            matches = zip([kps[m.queryIdx] for m in matches], [self.last['kps'][m.trainIdx] for m in matches])


        self.last = {'kps' : kps, 'des' : des}
        return matches
