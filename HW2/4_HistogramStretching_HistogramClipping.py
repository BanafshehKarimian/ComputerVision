import cv2
import numpy as np

def calc (image2,b_min,g_min,r_min,n):
    b_max = int(bb[-n])
    g_max = int(gg[-n])
    r_max = int(rr[-n])
    b,g,r=cv2.split(image2)
    for i in range(len(b)):
        for j in range(len(b[i])):
            b[i][j]=(b[i][j]-b_min)*255/(b_max-b_min)
            g[i][j]=(g[i][j]-g_min)*255/(g_max-g_min)
            r[i][j]=(r[i][j]-r_min)*255/(r_max-r_min)
    res = np.zeros((b.shape[0], b.shape[1], 3))
    res[:, :, 0] = b
    res[:, :, 1] = g
    res[:, :, 2] = r
    return res

image1 = cv2.imread('1.png',1)
b,g,r = cv2.split(image1)
b_min=b.min()
b_max=b.max()
g_min=g.min()
g_max=g.max()
r_min=r.min()
r_max=r.max()
for i in range(len(b)):
    for j in range(len(b[i])):
        b[i][j]=(b[i][j]-b_min)*255/(b_max-b_min)
        g[i][j]=(g[i][j]-g_min)*255/(g_max-g_min)
        r[i][j]=(r[i][j]-r_min)*255/(r_max-r_min)
img = np.zeros((b.shape[0], b.shape[1], 3))
img [:,:,0] = b
img [:,:,1] = g
img [:,:,2] = r
####################################P2
cv2.imwrite('stretched_img.png',img)
image2 = cv2.imread('2.jpg',1)
b,g,r = cv2.split(image2)
bb=b.flatten()
bb.sort()
gg=g.flatten()
gg.sort()
rr=r.flatten()
rr.sort()
b_min=b.min()
g_min=g.min()
r_min=r.min()
res1=calc(image2,b_min,g_min,r_min,1)
res2=calc(image2,b_min,g_min,r_min,2)
res5=calc(image2,b_min,g_min,r_min,5)
res10=calc(image2,b_min,g_min,r_min,10)
cv2.imwrite('1percent.png',res1)
cv2.imwrite('2percent.png',res2)
cv2.imwrite('5percent.png',res5)
cv2.imwrite('10percent.png',res10)
#print(res1)
#print("cccccccccccc")
#print(res10)