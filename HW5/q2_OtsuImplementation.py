import cv2
import numpy as np
from matplotlib import pyplot as plt
thresholds=[]
def otsu(gray):
    his, _ = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in range(255):
        value = np.sum(his[t:])*np.sum(his[:t])* ((1.0/(gray.shape[0] * gray.shape[1]))**2)*((np.mean(his[:t]) - np.mean(his[t:])) ** 2)
        if value > final_value:
            final_thresh = t
            final_value = value
        thresholds.append(value)
        plt.bar(t, value, .1, alpha=0.5, color='navy')
    for i in range(len(gray)):
        for j in range(len(gray[0])):
            if gray[i][j] > final_thresh:
                gray[i][j] = 255
            else:
                gray[i][j] = 0
    return gray,final_thresh

img = cv2.imread('redBall.png',0)
fin,v= otsu(img)
cv2.imwrite("fin.jpg",fin)
print(v)
y_pos = np.arange(len(thresholds))
plt.xticks(y_pos, thresholds)
plt.legend(loc='best')
plt.show()