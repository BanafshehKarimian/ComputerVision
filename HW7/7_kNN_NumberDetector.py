import struct
import numpy as np
import cv2


def __convert_to_one_hot(vector, num_classes):
    result = np.zeros(shape=[len(vector), num_classes])
    result[np.arange(len(vector)), vector] = 1
    return result

def __resize_image(src_image, dst_image_height, dst_image_width):
    src_image_height = src_image.shape[0]
    src_image_width = src_image.shape[1]

    if src_image_height > dst_image_height or src_image_width > dst_image_width:
        height_scale = dst_image_height / src_image_height
        width_scale = dst_image_width / src_image_width
        scale = min(height_scale, width_scale)
        img = cv2.resize(src=src_image, dsize=(0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    else:
        img = src_image

    img_height = img.shape[0]
    img_width = img.shape[1]

    dst_image = np.zeros(shape=[dst_image_height, dst_image_width], dtype=np.uint8)

    y_offset = (dst_image_height - img_height) // 2
    x_offset = (dst_image_width - img_width) // 2

    dst_image[y_offset:y_offset+img_height, x_offset:x_offset+img_width] = img

    return dst_image

def read_hoda_cdb(file_name):
    with open(file_name, 'rb') as binary_file:

        data = binary_file.read()

        offset = 0

        # read private header

        yy = struct.unpack_from('H', data, offset)[0]
        offset += 2

        m = struct.unpack_from('B', data, offset)[0]
        offset += 1

        d = struct.unpack_from('B', data, offset)[0]
        offset += 1

        H = struct.unpack_from('B', data, offset)[0]
        offset += 1

        W = struct.unpack_from('B', data, offset)[0]
        offset += 1

        TotalRec = struct.unpack_from('I', data, offset)[0]
        offset += 4

        LetterCount = struct.unpack_from('128I', data, offset)
        offset += 128 * 4

        imgType = struct.unpack_from('B', data, offset)[0]  # 0: binary, 1: gray
        offset += 1

        Comments = struct.unpack_from('256c', data, offset)
        offset += 256 * 1

        Reserved = struct.unpack_from('245c', data, offset)
        offset += 245 * 1

        if (W > 0) and (H > 0):
            normal = True
        else:
            normal = False

        images = []
        labels = []

        for i in range(TotalRec):

            StartByte = struct.unpack_from('B', data, offset)[0]  # must be 0xff
            offset += 1

            label = struct.unpack_from('B', data, offset)[0]
            offset += 1

            if not normal:
                W = struct.unpack_from('B', data, offset)[0]
                offset += 1

                H = struct.unpack_from('B', data, offset)[0]
                offset += 1

            ByteCount = struct.unpack_from('H', data, offset)[0]
            offset += 2

            image = np.zeros(shape=[H, W], dtype=np.uint8)

            if imgType == 0:
                # Binary
                for y in range(H):
                    bWhite = True
                    counter = 0
                    while counter < W:
                        WBcount = struct.unpack_from('B', data, offset)[0]
                        offset += 1
                        # x = 0
                        # while x < WBcount:
                        #     if bWhite:
                        #         image[y, x + counter] = 0  # Background
                        #     else:
                        #         image[y, x + counter] = 255  # ForeGround
                        #     x += 1
                        if bWhite:
                            image[y, counter:counter + WBcount] = 0  # Background
                        else:
                            image[y, counter:counter + WBcount] = 255  # ForeGround
                        bWhite = not bWhite  # black white black white ...
                        counter += WBcount
            else:
                # GrayScale mode
                data = struct.unpack_from('{}B'.format(W * H), data, offset)
                offset += W * H
                image = np.asarray(data, dtype=np.uint8).reshape([W, H]).T

            images.append(image)
            labels.append(label)

        return images, labels

def read_hoda_dataset(dataset_path, images_height=32, images_width=32, one_hot=False, reshape=True):
    images, labels = read_hoda_cdb(dataset_path)
    assert len(images) == len(labels)

    X = np.zeros(shape=[len(images), images_height, images_width], dtype=np.float32)
    Y = np.zeros(shape=[len(labels)], dtype=np.int)

    for i in range(len(images)):
        image = images[i]
        # Image resizing.
        image = __resize_image(src_image=image, dst_image_height=images_height, dst_image_width=images_width)
        # Image normalization.
        image = image / 255
        # Image binarization.
        image = np.where(image >= 0.5, 1, 0)
        # Image.
        X[i] = image
        # Label.
        Y[i] = labels[i]

    if one_hot:
        Y = __convert_to_one_hot(Y, 10).astype(dtype=np.float32)
    else:
        Y = Y.astype(dtype=np.float32)

    if reshape:
        X = X.reshape(-1, images_height * images_width)
    else:
        X = X.reshape(-1, images_height, images_width, 1)
    return X, Y

def features(img):
    _, cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(cnt[0])
    aspect_ratio = float(w) / h
    area = cv2.contourArea(cnt[0])
    equi_diameter = np.sqrt(4 * area / np.pi)
    rect_area = w * h
    extent = float(area) / rect_area
    #(_,_), (MA, ma), angle = cv2.fitEllipse(cnt[0])
    cnt = cnt[0]
    leftmostx,leftmosty = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmostx,rightmosty = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmostx,topmosty = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommostx,bottommosty = tuple(cnt[cnt[:, :, 1].argmax()][0])
    return  aspect_ratio,equi_diameter#,extent,leftmostx,leftmosty,rightmostx,rightmosty,topmostx,topmosty,bottommostx,bottommosty
'''
def nearest(train_features,train_labels,feature):
    min = 10000000000000
    label = 0
    for i in range(len(train_features)):
        x=0
        for j in range(len(feature)):
            x = x+ (train_features[i][j]-feature[j])**2
        if x < min:
            min = x
            label = train_labels[i]
    return label
'''
def knn(k,train_features,train_labels,feature):
    n = []
    label = []
    for i in range(k):
        x = 0
        for j in range(len(feature)):
            x = x + (train_features[i][j] - feature[j]) ** 2
        n.append(x)
        label.append(train_labels[i])
    for i in range(len(train_features)):
        x=0
        for j in range(len(feature)):
            x = x+ (train_features[i][j]-feature[j])**2
        min = np.min(n)
        if x < min:
            for l in range(k):
                if n[l]==min:
                    label[l] = train_labels[i]
                    n[l]=x
    (values, counts) = np.unique(label, return_counts=True)
    ind = np.argmax(counts)
    return values[ind]

train_images, train_labels = read_hoda_cdb('./DigitDB/Train 60000.cdb')
test_images, test_labels = read_hoda_cdb('./DigitDB/Test 20000.cdb')
train_features=[]
#calculate Features
for i in train_images:
    f = features(i)
    train_features.append(f)
test_features=[]
for i in test_images:
    f = features(i)
    test_features.append(f)
acc = 0
for i in range(len(test_labels)-18000):
    if knn(5,train_features,train_labels,test_features[i])==test_labels[i] :
        acc = acc +1
print(acc/2000)
