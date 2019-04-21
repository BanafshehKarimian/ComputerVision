import cv2
import numpy as np

def Line_Seg_D(img):
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(img)[0]
    return  lines

def Hough_line():
    img = cv2.imread('rectangles.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 100, 200, apertureSize=3)


    minLineLength = 30
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 15, minLineLength, maxLineGap)
    return lines

img = cv2.imread('rectangles.jpg',0)
lines = Line_Seg_D(img)
line_img = np.zeros([img.shape[0],img.shape[1],3])
for i in range(len(lines)):
    x1,y1,x2,y2= lines[i][0]
    cv2.line(line_img, (x1, y1), (x2,y2), (0,255,0), 3)

lines2 = Hough_line()
line_img2 = np.zeros([img.shape[0],img.shape[1],3])
for i in range(len(lines)):
    x1,y1,x2,y2= lines2[i][0]
    cv2.line(line_img2, (x1, y1), (x2,y2), (0,255,0), 3)

cv2.imwrite("Q5-lines-LSD.jpg",line_img)
cv2.imwrite("Q5-lines-Hough.jpg",line_img2)
