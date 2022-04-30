#!/usr/bin/python3

import cv2
import pygame
import time
import sdl2.ext

W = 1920//2
H = 1080//2

pygame.init()
display = pygame.display.set_mode((W,H))
surface = pygame.Surface((W,H)).convert()

def process_frame(img):
    img = cv2.resize(img, (W, H))
    pygame.surfarray.blit_array(surface, img.swapaxes(0,1))
    display.blit(surface, (0,0))
    pygame.display.update()
    pygame.display.flip()

if __name__ == "__main__":
    cap = cv2.VideoCapture("test1.mp4")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret == True:
            process_frame(frame)
        else:
            break

