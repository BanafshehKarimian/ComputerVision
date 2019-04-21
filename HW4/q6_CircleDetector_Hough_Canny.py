import cv2
import numpy as np

img = cv2.imread('Balls.jpg',0)
cimg = cv2.imread('Balls.jpg')

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=50,maxRadius=100)
c_img = np.zeros([img.shape[0],img.shape[1],3])
print(circles[0])
for i in range(len(circles[0])):
    [x,y,r]= circles[0][i]
    cv2.circle(c_img, (x, y), r, (0, 0, 255), 0)

edges = cv2.Canny(img, 70, 200, apertureSize=3)

edges2 = cv2.Canny(img, 0, 200, apertureSize=3)
circles2 = cv2.HoughCircles(edges2,cv2.HOUGH_GRADIENT,1,100,param1=50,param2=30,minRadius=10,maxRadius=200)
c_img2 = np.zeros([img.shape[0],img.shape[1],3])
print(circles2[0])
for i in range(len(circles2[0])):
    [x,y,r]= circles2[0][i]
    cv2.circle(c_img2, (x, y), r, (0, 0, 255), 0)

for i in range(len(circles2[0])):
    [x,y,r]= circles2[0][i]
    cv2.circle(cimg, (x, y), r, (0, 0, 255), 2)

cv2.imwrite("Q6-Hough.jpg",c_img)
cv2.imwrite("Q6-Canny.jpg",edges)
cv2.imwrite("Q6-HoughCannyMerged.jpg",c_img2)
cv2.imwrite("Q6-final.jpg",cimg)