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

    def extract(self, img):
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

        feats = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)    # strong corners on an image
        print(feats)
        return feats


fe = FeatureExtractor()

def process_image(img):
    img = cv2.resize(img, (W,H))
    kp = fe.extract(img)

    for f in kp:
        print(f)

    for f in kp:
        u, v = map(lambda x: int(round(x)),f[0])
        cv2.circle(img, (u,v), color=(0,255,0), radius=3)

    display.draw(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("video/test1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_image(frame)
        else:
            break


