import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('striping.bmp',0)

dft = cv2.dft(np.float32(img),flags = cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

for i in range(len(dft_shift)):
    for j in range(len(dft_shift[i])):
        if (j-399)**2+(i-624)**2<400 or (j-933)**2+(i-707)**2<400:
            dft_shift[i][j][0]=dft_shift[i][j][1]=0.1

magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))


cv2.imwrite('out.png',magnitude_spectrum)

rows, cols = img.shape
crow,ccol = rows/2 , cols/2

mask = np.zeros((rows,cols,2),np.uint8)
mask[int(crow)-30:int(crow)+30, int(ccol)-30:int(ccol)+30] = 1

fshift = dft_shift*mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:,:,0],img_back[:,:,1])

plt.subplot(121),plt.imshow(img_back, cmap = 'gray')
plt.title('Output Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Fixed Fourie'), plt.xticks([]), plt.yticks([])
plt.show()