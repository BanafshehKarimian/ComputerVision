import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

def tmp(img,name):
    img2 = img.copy()
    img3 = img.copy()
    template = cv2.imread('Flag/template.jpg', 0)
    w, h = template.shape[::-1]
    res1 = []
    res2 = []
    res3 = []
    #print(np.shape(img))
    i= h-50
    j= w-50
    v1=0
    v2=0
    v3=100000000000
    t1 = 0
    t2 = 0
    t3 = 0
    while i < len(img[0])-60:
        i = i + 50
        #print('c')
        while j < len(img)-60:
            #print(str(i)+'+'+str(j))
            j = j + 50
            res_template = cv2.resize(template, (i, j))
            y = time.time()
            res = cv2.matchTemplate(img, res_template, cv2.TM_CCOEFF_NORMED)
            t1 = t1 + time.time()-y
            _,x,_,_=cv2.minMaxLoc(res)
            if(x>v1):
                v1=x
                res1= res
                w1, h1 = res_template.shape[::-1]
            y = time.time()

            res = cv2.matchTemplate(img2, res_template, cv2.TM_CCORR_NORMED)
            t2 = t2 + time.time() - y

            _, x, _, _ = cv2.minMaxLoc(res)
            if (x > v2):
                v2 = x
                res2 = res
                w2, h2 = res_template.shape[::-1]
            y = time.time()

            res = cv2.matchTemplate(img3, res_template, cv2.TM_SQDIFF_NORMED)
            t3 = t3 + time.time() - y

            x, _, _, _ = cv2.minMaxLoc(res)
            if (x < v3):
                v3 = x
                res3=res
                w3, h3 = res_template.shape[::-1]

    i = h
    j = w

    while i > 50:
        i = i - 50
        #print('c')
        while j > 50:
            #print(str(i)+'+'+str(j))
            j = j - 50
            res_template = cv2.resize(template, (i, j))
            y = time.time()

            res = cv2.matchTemplate(img, res_template, cv2.TM_CCOEFF_NORMED)
            t1 = t1 + time.time() - y

            _,x,_,_=cv2.minMaxLoc(res)
            if(x>v1):
                v1=x
                res1= res
                w1, h1 = res_template.shape[::-1]
            y = time.time()

            res = cv2.matchTemplate(img2, res_template, cv2.TM_CCORR_NORMED)
            t2 = t2 + time.time() - y

            _, x, _, _ = cv2.minMaxLoc(res)
            if (x > v2):
                v2 = x
                res2 = res
                w2, h2 = res_template.shape[::-1]
            y = time.time()

            res = cv2.matchTemplate(img3, res_template, cv2.TM_SQDIFF_NORMED)
            t3 = t3 + time.time() - y

            x, _, _, _ = cv2.minMaxLoc(res)
            if (x < v3):
                v3 = x
                res3=res
                w3, h3 = res_template.shape[::-1]


    #print('finished')
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res1)
    top_left = max_loc
    bottom_right = (top_left[0] + w1, top_left[1] + h1)
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imwrite(name + 'COFF.png', img)
    print("COEFF ACC:" + str(max_val) + "  Time:" + str(t1))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res2)
    top_left = max_loc
    bottom_right = (top_left[0] + w2, top_left[1] + h2)
    cv2.rectangle(img2, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imwrite(name + 'CORR.png', img2)
    print("CORR ACC:"+str(max_val) + "  Time:" + str(t2))

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res3)
    top_left = min_loc
    bottom_right = (top_left[0] + w3, top_left[1] + h3)
    cv2.rectangle(img3, top_left, bottom_right, (0, 0, 255), 2)
    cv2.imwrite(name + 'SQDIFF.png', img3)
    print("SQDIFF ACC:"+str(min_val) +"  Time:"+ str(t3))


image1 = cv2.imread('Flag/1.jpg',0)
image2 = cv2.imread('Flag/2.jpg',0)
image3 = cv2.imread('Flag/3.jpg',0)
image4 = cv2.imread('Flag/4.jpg',0)

tmp(image1,"img1")
tmp(image2,"img2")
tmp(image3,"img3")
tmp(image4,"img4")
