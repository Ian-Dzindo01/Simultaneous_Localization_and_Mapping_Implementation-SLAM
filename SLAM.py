# !/usr/bin/env python
import cv2
import pygame
import time
import sdl2.ext
sdl2.ext.init()

W = 1920//2
H = 1080//2

class Display(object):
    def __init__(self, W, H):
        sdl2.ext.init()

        self.W, self.H = W, H
        self.window = sdl2.ext.Window("SLAM", size=(W,H), position=(-500,-500))
        self.window.show()

    def paint(self, img):
        events = sdl2.ext.get_events()
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        surf = sdl2.ext.pixels3d(self.window.get_surface())                        # creates a 2D numpy array from the given Surface object
        surf[:] = img.swapaxes(0, 1)
        self.window.refresh()
        cv2.imshow('image', img)

Display(W,H)

def process_frame(img):
    img = cv2.resize(img, (W,H))

if __name__ == "__main__":
    cap = cv2.VideoCapture("test1.mp4")

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break
