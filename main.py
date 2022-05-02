#!/usr/bin/python3

import cv2
import time
from display import Display
import numpy as np


W = 1920//2
H = 1080//2

display = Display(W, H)

class FeatureExtractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(100)   # 1000 nfeatures
        self.bf = cv2.BFMatcher()        # brute force matcher
        self.last = None

    def extract(self, img):

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)    # strong corners on an image

        # extraction
        kps = [cv2.KeyPoint(x=f[0][0], y=f[0][1], size=20) for f in feats]
        kps, des = self.orb.compute(img, kps)

        if self.last is not None:
            matches = self.bf.match(des, self.last['des'])
            print(matches)

        self.last = {'kps' : kps, 'des' : des}

        return kps, des

fe = FeatureExtractor()

def process_image(img):
    img = cv2.resize(img, (W,H))
    kps, des = fe.extract(img)

    for p in kps:
        u, v = map(lambda x: int(round(x)), p.pt)            # extract feature locations
        cv2.circle(img, (u,v), color=(0,255,0), radius=3)    # draw circle around points

    display.draw(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("video/test1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_image(frame)
        else:
            break

# # run detect in grid
# sy = img.shape[0]//self.GX
# sx = img.shape[1]//self.GX

# akp = []
# for ry in range(0, img.shape[0], sy):
#     for rx in range(0, img.shape[1], sx):
#         img_block = img[ry:ry+sy, rx:rx+sx]           # working in a grid range
#         kp = self.orb.detect(img_block, None)         # keypoints and descriptors

#         for p in kp:
#             p.pt = (p.pt[0] + rx, p.pt[1] + ry)
#             akp.append(p)

# return akp
