#!/usr/bin/python3

import cv2
import time
import numpy as np
from display import Display
from featureExtractor import FeatureExtractor


W = 1920//2
H = 1080//2

F = 1

K = np.array(([F,0,W//2], [0,F,H//2], [0,0,1]))    # intrinsic matrix

display = Display(W, H)
fe = FeatureExtractor(K)

def process_image(img):
    img = cv2.resize(img, (W,H))    # image resize to W, H
    matches = fe.extract(img)       # extract features wth FE

    print("%d matches" % (len(matches)))

    for pt1, pt2 in matches:
        u1,v1 = fe.denormalize(pt1)
        u2,v2 = fe.denormalize(pt2)
        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)      # draw circles aroung the matches
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))        # draw lines in between

    display.draw(img)        # display image


# video capture
if __name__ == "__main__":
    cap = cv2.VideoCapture("video/test2.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_image(frame)
        else:
            break
