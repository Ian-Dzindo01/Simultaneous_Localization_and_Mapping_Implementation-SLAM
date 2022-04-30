#!/usr/bin/python3

import cv2
import pygame
import time
import sdl2.ext

sdl2.ext.init()

W = 1920//2
H = 1080//2

def process_frame(img):
    img = cv2.resize(img, (W, H))
    events = sdl2.ext.get_events()
    cv2.imshow('image', img)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_frame(frame)
        else:
            break

