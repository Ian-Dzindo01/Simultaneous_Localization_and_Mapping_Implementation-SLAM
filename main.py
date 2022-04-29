#!/usr/bin/python3

import cv2
import pygame
import time
import sdl2.ext
W = 1920//2
H = 1080//2

screen = pygame.display.set_mode((W,H))

cv2.namedWindow('image', cv2.WINDOW_NORMAL)

def process_image(img):
    img = cv2.resize(img, (1920//2, 1080//2))
    pygame.display.flip()
    # cv2.imshow('image', img)
    print(img.shape)

if __name__ == "__main__":
    cap = cv2.VideoCapture("test1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()
        # print(ret)
        if ret == True:
            process_image(frame)
        else:
            break

