import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

def svm(X_test,X_train,y_train):
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X_train, y_train)
    y_pred = svclassifier.predict(X_test)
    return y_pred


def get_HOG(img):
    hog = cv2.HOGDescriptor()
    h = hog.compute(img)
    return h
def edge_box(img,model):
    edge_detection = cv2.ximgproc.createStructuredEdgeDetection(model)
    rgb_im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

    orimap = edge_detection.computeOrientation(edges)
    edges = edge_detection.edgesNms(edges, orimap)

    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(30)
    boxes = edge_boxes.getBoundingBoxes(edges, orimap)

    for b in boxes:
        x, y, w, h = b
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow("edges", edges)
    cv2.imshow("edgeboxes", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sliding_window(image,stepSize = 50):
    tmp = image  # for drawing a rectangle
    (w_width, w_height) = (50, 50)  # window size
    for x in range(0, image.shape[1] - w_width, stepSize):
        for y in range(0, image.shape[0] - w_height, stepSize):
            window = image[x:x + w_width, y:y + w_height, :]
            #....
