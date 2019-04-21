import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
def B(s):
    n = 5
    img = cv2.imread(s,0)
    F = np.fft.fftshift(np.fft.fft2(img))
    H = np.copy(F)
    G = np.copy(F)
    for i in range(len(H)):
        for j in range(len(H)):
            H[i][j]= 1/(1+((abs(i-128))**(2*n)+(abs(j-128))**(2*n))/30**(2*n))
            G[i][j] = F[i][j]*H[i][j]
    return np.abs(np.fft.ifft2(np.fft.ifftshift(G)))

img_back = B('1.jpg')
img_back2 = B('2.jpg')

cv2.imwrite('2output1.png',img_back)

cv2.imwrite('2output2.png',img_back2)

