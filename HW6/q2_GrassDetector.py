import cv2
import numpy as np


def stadium(img,s1,s2):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    std = cv2.inRange(hsv, (36, 0, 0), (70, 255, 255))

    kernel = np.ones((s1, s1), np.uint8)

    closing = cv2.morphologyEx(std, cv2.MORPH_CLOSE, kernel)
    kernel = np.ones((s2,s2), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    return opening


img = cv2.imread('Stadium1.jpg',1)
opening = stadium(img,10,85)
img = cv2.imread('Stadium2.png',1)
opening2 = stadium(img,10,30)
cv2.imshow('stadium',opening)
cv2.imwrite('2-1.png',opening)
cv2.imshow('stadium2',opening2)
cv2.imwrite('2-2.png',opening2)
cv2.waitKey(0)
cv2.distroyAllWindows()