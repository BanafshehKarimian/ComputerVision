import cv2
import numpy as np

def labels(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, labels = cv2.connectedComponents(img_gray)
    return labels
def connected_sep(labels):
    cnn=[]
    all = []
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            if labels[i][j] not in cnn:
                cnn.append(labels[i][j])
    for lb in cnn:
        x = np.zeros((len(labels),len(labels[0])))
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i][j]==lb:
                    x[i][j]=255
        all.append(x)
    return all
def connected(img):
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img_gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    binary_map = (img_gray > 0).astype(np.uint8)
    connectivity = 4 
    output,_,_,_ = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    print("connected Number:")
    print(output)
    return output,closing

def rects(img):
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img_gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)
    binary_map = (img_gray > 0).astype(np.uint8)
    connectivity = 4  
    output,_,_,_ = cv2.connectedComponentsWithStats(binary_map, connectivity, cv2.CV_32S)
    print("connected Number:")
    print(output)
    return output,closing

def holes(img):
    kernel = np.ones((10, 10), np.uint8)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    for i in range(len(img)):
        for j in range(len(img[0])):
            closing[i][j]=closing[i][j]-img[i][j]

    kernel = np.ones((7,7), np.uint8)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    img_gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    connectivity = 4 
    output,_,_,_ = cv2.connectedComponentsWithStats(img_gray, connectivity, cv2.CV_32S)
    return output,opening
def counter(comp,hole):
    count = np.zeros(len(comp))
    for i in range(len(comp)):
        for j in range(len(hole)):
            x = np.sum(comp[i]*hole[j])
            if x!= 0:
                count[i]=count[i]+1
        count[i]= count[i]-1
    hole1=0
    hole2=0
    for i in count:
        if i !=0 :
          if i ==1:
              hole1 = hole1+1
          else:
              hole2 = hole2+1
    return hole1,hole2


img = cv2.imread('shapes.png',1)
cnnum,cnn = connected(img)
components = connected_sep(labels(cnn))
hlnum,hl = holes(img)
hol=connected_sep(labels(hl))
h1,h2=counter(components,hol)
print("number of components with one hole:")
print(h1)
print("number of components with two holes:")
print(h2)
print("number of holes:")
print(h1+2*h2)
kernel = np.zeros((30, 30), np.int8)
cl = cv2.cvtColor(cnn, cv2.COLOR_BGR2GRAY)
k=0
#Count Number of squars using this pattern
#0000
#0111
#01
#01
for i in range(len(cl)-10):
    for j in range(len(cl[i]) - 10):
        w=0
        for l in range(10):
            if cl[i+l][j]==0 and cl[i+l][j+1]>0 :
                w = w+1
            if cl[i][j+ l]==0 and cl[i+1][j+l]> 0:
                w = w + 1
        if w > 15 :
            k = k+1
print("Number of squars:")
print(k)
print("Nuber of circles:")
print(cnnum-k)
cv2.imshow('holes',cl)
cv2.waitKey(0)
cv2.distroyAllWindows()