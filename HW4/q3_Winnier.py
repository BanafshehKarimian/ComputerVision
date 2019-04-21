import numpy as np
import matplotlib.pyplot as plt
from skimage import color, data, restoration
from scipy.signal import convolve2d as conv2
import cv2

def winier(k,s):
    h = cv2.imread('h.jpg', 0)
    x = np.fft.fft2(h)
    H = np.fft.fftshift(x)
    Hc = np.conj(H)
    f = cv2.imread(s, 0)
    xx = np.fft.fft2(f)
    F = np.fft.fftshift(xx)
    Res = np.copy(F)
    for i in range(len(Res)):
        for j in range(len(Res)):
            Res[i][j] = F[i][j] * (Hc[i][j] / (np.real(H[i][j]) ** 2 + np.imag(H[i][j]) ** 2 + k))
    f_ishift = np.fft.ifftshift(Res)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back


k=10

img1w= winier(k,'degeraded1.jpg')

img2w= winier(k,'degeraded2.jpg')

img1rev= winier(0.00000000001,'degeraded1.jpg')

img2rev= winier(0.00000000001,'degeraded2.jpg')

plt.subplot(141),plt.imshow(img1w, cmap = 'gray')
plt.title('1winier'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(img1rev, cmap = 'gray')
plt.title('1reverse'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(img2w, cmap = 'gray')
plt.title('2winier'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img2rev, cmap = 'gray')
plt.title('2reverse'), plt.xticks([]), plt.yticks([])

plt.show()
