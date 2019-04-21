import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('striping.bmp',0)
f = np.fft.fft2(img)
fshift = np.fft.fftshift(f)


m1 = np.log(abs(fshift))


for i in range(len(fshift)):
    for j in range(len(fshift[i])):
        if (j-399)**2+(i-624)**2<400 or (j-933)**2+(i-707)**2<400:
            fshift[i][j]=0.1

m2 = np.log(abs(fshift))

f_ishift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)

cv2.imwrite('output.png',img_back)

plt.subplot(141),plt.imshow(img, cmap = 'gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(142),plt.imshow(m1, cmap = 'gray')
plt.title('Fourier'), plt.xticks([]), plt.yticks([])
plt.subplot(143),plt.imshow(m2, cmap = 'gray')
plt.title('fixed Fourier'), plt.xticks([]), plt.yticks([])
plt.subplot(144),plt.imshow(img_back, cmap = 'gray')
plt.title('OutputImage'), plt.xticks([]), plt.yticks([])

plt.show()