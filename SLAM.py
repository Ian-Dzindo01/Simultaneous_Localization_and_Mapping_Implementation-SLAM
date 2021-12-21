import cv2
import pygame

W = 1920//2
H = 1080//2

pygame.init()
screen = pygame.display.set_mode((W, H))

def process_frame(img):
    img = cv2.resize(img, (W, H))
    pygame.display.flip()
    print(img.shape)

if __name__ == '__main__':
    cap = cv2.VideoCapture("test1.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            process_frame(frame)
        else:
            break

