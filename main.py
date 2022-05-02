#!/usr/bin/python3

import cv2
import time
from display import Display
from featureExtractor import FeatureExtractor

W = 1920//2
H = 1080//2

display = Display(W, H)
fe = FeatureExtractor()

def process_image(img):
    img = cv2.resize(img, (W,H))    # image resize to W, H
    matches = fe.extract(img)       # extract features wth FE

    for pt1, pt2 in matches:
        u1,v1 = map(lambda x: int(round(x)), pt1.pt)         # unzip match data
        u2,v2 = map(lambda x: int(round(x)), pt2.pt)

        cv2.circle(img, (u1, v1), color=(0,255,0), radius=3)      # draw circles aroung the matches
        cv2.line(img, (u1, v1), (u2, v2), color=(255,0,0))        # draw lines in between

    display.draw(img)        # display image


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

# for p in kps:
#     u, v = map(lambda x: int(round(x)), p.pt)            # extract feature locations
#     cv2.circle(img, (u,v), color=(0,255,0), radius=3)    # draw circle around points
