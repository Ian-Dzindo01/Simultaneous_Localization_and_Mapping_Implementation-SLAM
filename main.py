#!/usr/bin/python3

import cv2
import time
from display import Display

W = 1920//2
H = 1080//2

display = Display(W, H)

class FeatureExtractor(object):
    GX = 16//2
    GY = 12//2

    def __init__(self):
        self.orb = cv2.ORB_create(1000)   # 1000 nfeatures

    def extract(self, img):
        # run detect in grid
        sy = img.shape[0]//self.GX
        sx = img.shape[1]//self.GX

        akp = []
        for ry in range(0, img.shape[0], sy):
            for rx in range(0, img.shape[1], sx):
                img_block = img[ry:ry+sy, rx:rx+sx]           # working in a grid range
                kp = self.orb.detect(img_block, None)         # keypoints and descriptors

                for p in kp:
                    p.pt = (p.pt[0] + rx, p.pt[1] + ry)
                    akp.append(p)

        return akp



fe = FeatureExtractor()

def process_image(img):
    img = cv2.resize(img, (W,H))
    kp = fe.extract(img)

    for p in kp:
        u, v = map(lambda x: int(round(x)), p.pt)
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


