#!/usr/bin/python3

import cv2
import time
from display import Display

W = 1920//2
H = 1080//2

display = Display(W, H)
orb = cv2.ORB_create()

def process_image(img):
    img = cv2.resize(img, (W,H))

    kp, des = orb.detectAndCompute(img, None)        #keypoints and descriptors
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


