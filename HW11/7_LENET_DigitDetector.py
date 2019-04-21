from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD

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


def LeNet(train_data, train_labels,test_data, test_labels):
    model = Sequential()
    #1
    model.add(Convolution2D(
        filters=20,
        kernel_size=(6, 5),
        padding="same",
        input_shape=(32,32,1)))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))
    #2
    model.add(Convolution2D(
        filters=50,
        kernel_size=(16, 5),
        padding="same"))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))

    model.add(Flatten())
    #3
    model.add(Dense(120))

    model.add(Activation(
        activation="relu"))
    #4
    model.add(Dense(84))

    model.add(Activation(
        activation="relu"))
    #5
    model.add(Dense(10))

    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.01),
        metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        batch_size=128,
        nb_epoch=5)

    (loss, accuracy) = model.evaluate(
        test_data,
        test_labels,
        batch_size=128,
        verbose=1
    )

    return loss,accuracy


def Net(train_data, train_labels,test_data, test_labels,filter_num,kernel_s,d,pad,pool_s,stride_s,conv_num,fully_num):

    model = Sequential()
    model.add(Convolution2D(
        filters=filter_num[0],
        kernel_size=kernel_s[0],
        padding=pad,
        input_shape=(32,32,1)))

    for i in range(conv_num-1):
        model.add(Convolution2D(
        filters=filter_num[i],
        kernel_size=kernel_s[i],
        padding="valid"))
        model.add(MaxPooling2D(
            pool_size=pool_s,
            strides=stride_s))
    model.add(Flatten())
    for i in range(fully_num-1):
        model.add(Dense(d[i]))
        model.add(Activation("relu"))

    model.add(Dense(10))
    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.01),
        metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        batch_size=128,
        nb_epoch=5,
        verbose=1)

    (loss, accuracy) = model.evaluate(
        test_data,
        test_labels,
        batch_size=128,
        verbose=1
    )

    return loss,accuracy


#print("data Ready")
#train_data=train_images.reshape((train_images.shape[0], 27, 20))
#test_data=test_images.reshape((test_images.shape[0], 27, 20))
train_images, train_labels = read_hoda_dataset(dataset_path='./DigitDB/Train 60000.cdb',
                                images_height=32,
                                images_width=32,
                                one_hot=False,
                                reshape=False)
test_images, test_labels = read_hoda_dataset(dataset_path='./DigitDB/Test 20000.cdb',
                              images_height=32,
                              images_width=32,
                              one_hot=True,
                              reshape=False)
print("yess")
y = np.zeros([60000,10])

for i in range(60000):
    y[i][int(train_labels[i])]=1
train_labels = y


train_data = train_images
test_data = test_images
print(np.shape(train_images))
# Reshape the data to a (70000, 28, 28, 1) tensord
#train_data = train_data[:, :, :, np.newaxis]
#test_data = test_data[:, :, :, np.newaxis]
print("tttttttttttttt")
print(np.shape(train_labels))
print(train_labels)
#Net(train_data, train_labels,test_data, test_labels,filter_num,kernel_s,d,pad,pool_s,stride_s,conv_num,fully_num)
#print(LeNet(train_data, train_labels,test_data, test_labels))

def test(i):
    # test conv layer:
    if (i==1):
        print("testing for conv layer of 2 and 3")
        l, a = Net(train_images, train_labels, test_images, test_labels, [40, 40],
                   [(6, 5), (6, 5)], [120, 84], "valid", (2, 2), (2, 2), 2, 3)
        print(l, a)
        l, a = Net(train_images, train_labels, test_images, test_labels, [40, 40, 40], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
    # test dens layer:
    if (i==2):
        print("testing for dens layer of 3 ,5 and 10")
        l, a = Net(train_images, train_labels, test_images, test_labels, [40, 40, 40], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
        l, a = Net(train_images, train_labels, test_images, test_labels, [40, 40, 40], [(6, 5), (6, 5), (6, 5)],
                   [120, 84, 74, 64], "valid", (2, 2), (2, 2), 3, 5)
        print(l, a)
        l, a = Net(train_images, train_labels, test_images, test_labels, [40, 40, 40], [(6, 5), (6, 5), (6, 5)],
                   [120, 84, 74, 64, 54, 44, 34, 24, 14], "valid", (2, 2), (2, 2), 3, 10)
        print(l, a)
    # test filter:
    if(i==3):
        print("testing for filter 10,20 and 40")
        l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
        print("----------------------------------------------------------------------------------")
        l, a = Net(train_images, train_labels, test_images, test_labels, [20, 20, 20], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
        print("----------------------------------------------------------------------------------")
        l, a = Net(train_images, train_labels, test_images, test_labels, [40, 40, 40], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
        print("----------------------------------------------------------------------------------")
    if(i==4):
    # test kernel window:
        print("testing for kernel window 6,16 and 26")
        #l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
         #          "valid", (2, 2), (2, 2), 3, 3)
        #print(l, a)
        #l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (16, 5), (16, 5)],
        #           [120, 84], "valid", (2, 2), (2, 2), 3, 3)
        #print(l, a)
        #l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(26, 5), (26, 5), (26, 5)],
        #           [120, 84], "valid", (2, 2), (2, 2), 3, 3)
        #print(l, a)
    if(i==5):
        # test padding:
        print("testing for padding")
        l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
        l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "same", (2, 2), (2, 2), 3, 3)
        print(l, a)
    if(i==6):
        # test padding:
        print("testing for stide 1,2 and 4")
        l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (1, 1), 3, 3)
        print(l, a)
        l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
                   "valid", (2, 2), (2, 2), 3, 3)
        print(l, a)
#        l, a = Net(train_images, train_labels, test_images, test_labels, [10, 10, 10], [(6, 5), (6, 5), (6, 5)], [120, 84],
#                  "valid", (2, 2), (4, 4), 3, 3)
#        print(l, a)

    # Net(train_images, train_labels,test_images, test_labels,filter_num,kernel_s,[120,84],"valid",(2,2),(2,2),conv_num,3)

print("testing params")
#test(3)
for i in range(6):
    test(i+1)