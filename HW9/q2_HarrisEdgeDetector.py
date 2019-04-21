from scipy import signal as sig
import numpy as np
import cv2

window_size = 7
k = 0.05
th = 230#cv2.adaptiveThreshold(img2, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

def func(img):
    gray  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)   
	#thresholding
    for i in range(len(gray)):
        for j in range(len(gray[i])):
            if gray[i][j]<th:
                gray[i][j] = 0
            else:
                gray[i][j]=255
	#init Kernels and Is
	kernelx = np.array([[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]])
	kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = sig.convolve2d(gray, kernelx, mode='same')
    Iy = sig.convolve2d(gray, kernely, mode='same')
    Ixx = Ix ** 2
    Ixy = Iy * Ix
    Iyy = Iy ** 2   
    h, w = gray.shape
    offset = (int)(window_size / 2)
    for y in range((int)(offset),(int)(h - offset)):
        for x in range((int)(offset), (int)(w - offset)):
			#calculate Sobel for harris
            Sxx = np.sum(Ixx[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Syy = np.sum(Iyy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            Sxy = np.sum(Ixy[y - offset:y + 1 + offset, x - offset:x + 1 + offset])
            r = (Sxx * Syy) - (Sxy ** 2) - k * ((Sxx + Syy) ** 2)
            if r > 200:
				img[y][x] = [255, 0, 0]
    return img


img = cv2.imread('left04.jpg')
img_res1 = func(img)
img2 = cv2.imread('right04.jpg')
img_res2 = func(img2)
cv2.imshow("edge", img_res1)
cv2.imshow("edge2", img_res2)
cv2.waitKey(0)
cv2.destroyAllWindows()