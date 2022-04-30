#!/usr/bin/python3

import cv2
import pygame
import time
from display import Display

W = 1920//2
H = 1080//2

display = Display(W, H)

def process_image(img):
    img = cv2.resize(img, (W,H))
    display.draw(img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("video/test1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_image(frame)
        else:
            break


