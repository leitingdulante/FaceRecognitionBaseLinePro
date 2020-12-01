import tensorflow as tf
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

my_faces_path = 'faces'
other_faces_path = './lfw'
size = 224
# LABEL_SIZE = 2800
LABEL_SIZE = len(os.listdir(my_faces_path)) + len(os.listdir(other_faces_path))
print("LABEL_SIZE", LABEL_SIZE)
imgs = []
labs = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    try:
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                filename = path + '/' + filename

                img = cv2.imread(filename)

                top,bottom,left,right = getPaddingSize(img)
                # 将图片放大， 扩充图片边缘部分
                img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
                img = cv2.resize(img, (h, w))

                imgs.append(img)
                labs.append(path.split("/")[-1])
            else:
                readData(path + "/" + filename)
    except Exception as e:
        print(e)

def processLabel(size):
    lbe = LabelBinarizer()
    global labs
    labs = lbe.fit_transform(labs)
    size = size - len(labs[0])
    labs = np.pad(labs, [[0,0], [0,size]])
    return labs, lbe

def weightVariable(shape):
    init = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(init)

def biasVariable(shape):
    init = tf.random_normal(shape)
    return tf.Variable(init)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxPool(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def dropout(x, keep):
    return tf.nn.dropout(x, keep)


x = tf.placeholder(tf.float32, [None, size, size, 3])
y_ = tf.placeholder(tf.float32, [None, LABEL_SIZE])

keep_prob_5 = tf.placeholder(tf.float32)
keep_prob_75 = tf.placeholder(tf.float32)

def cnnLayer(softmaxSize):
    # 第一层
    W1 = weightVariable([3,3,3,32]) # 卷积核大小(3,3)， 输入通道(3)， 输出通道(32)
    b1 = biasVariable([32])
    # 卷积
    conv1 = tf.nn.sigmoid(conv2d(x, W1) + b1)
    # 池化
    pool1 = maxPool(conv1)
    # 减少过拟合，随机让某些权重不更新
    drop1 = dropout(pool1, keep_prob_5)

    # 第二层
    W2 = weightVariable([3,3,32,64])
    b2 = biasVariable([64])
    conv2 = tf.nn.sigmoid(conv2d(drop1, W2) + b2)
    pool2 = maxPool(conv2)
    drop2 = dropout(pool2, keep_prob_5)

    # 第三层
    W3 = weightVariable([3,3,64,64])
    b3 = biasVariable([64])
    conv3 = tf.nn.sigmoid(conv2d(drop2, W3) + b3)
    pool3 = maxPool(conv3)
    drop3 = dropout(pool3, keep_prob_5)

    # 全连接层  # 注意这种操作参数是制约的，小心设置，或者改为自动计算的模式，不设定死参数
    Wf1 = weightVariable([12*12*64, 512])
    bf1 = biasVariable([512])
    drop3_flat = tf.reshape(drop3, [-1, 12*12*64])
    dense1 = tf.nn.relu(tf.matmul(drop3_flat, Wf1) + bf1)
    dropf1 = dropout(dense1, keep_prob_75)

    # 全连接层2
    Wf = weightVariable([512, 32])
    bf = biasVariable([32])
    dense = tf.nn.relu(tf.matmul(dropf1, Wf) + bf)
    dropf = dropout(dense, keep_prob_75)

    # 输出层
    Wout = weightVariable([32,softmaxSize])
    bout = biasVariable([softmaxSize])
    #out = tf.matmul(dropf, Wout) + bout
    out = tf.add(tf.matmul(dropf, Wout), bout)
    return out, dense