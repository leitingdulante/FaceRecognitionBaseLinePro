import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(pro_dir)

import tensorflow as tf
import cv2
import numpy as np
import dlib
import pickle
import base64
from uploadr.faceFind import vgg19

class FacePredict:

    def __init__(self):
        pass

    def init(self, size=224, isForceCalCacheData=False):
        self.size = size
        self.facePath = pro_dir + '/uploadr/faceFind/faces'
        self.cachePath = pro_dir + '/uploadr/faceFind/featureCachePure'
        print(self.cachePath)
        self.sess = tf.Session()
        self.cacheData = {}
        self.input = tf.placeholder(tf.float32, [None, self.size, self.size, 3])
        self.detector = dlib.get_frontal_face_detector()
        self.vgg = vgg19.Vgg19()
        self.vgg.build(self.input)
        # 加载所有需要加载的候选集向量
        # 1.看看是否有缓存，有加载，无重新预测得到向量，并缓存文件
        if not isForceCalCacheData:
            try:
                with open(self.cachePath, 'rb') as file_holder:
                    self.cacheData = pickle.load(file_holder)
            except Exception as e:
                print(e)

        # if len(self.cacheData) < 1:
        #     self.cacheData = self.createCacheData()
        #     print("cacheData size ", len(self.cacheData))
        #     self.saveCache()
        return True

    def predict(self, img, isNeedTailOnlyFace=True, threshold=-1):
        # 最低阈值距离要求，取值范围
        embMap, idxsMap = self.modelPredict({"def": img}, isNeedTailOnlyFace)
        if len(embMap) < 1:
            return None, None, None
        emb = list(embMap.values())[0]
        idxs = list(idxsMap.values())[0]
        simValue = 10000
        simKey = None
        for key, vector in self.cacheData.items():
            newDis = self.dis(emb, vector)
            if newDis < simValue:
                isSet = True
                if threshold != -1 and newDis > threshold:
                    isSet = False
                if isSet:
                    simValue = newDis
                    simKey = key
        return simKey, simValue, idxs

    def recordFace(self, name, img, isNeedTailOnlyFace=True):
        # 录入人脸数据
        embMap, idxsMap = self.modelPredict({"def": img}, isNeedTailOnlyFace)
        if len(embMap) < 1 or name is None:
            return False, ""
        emb = list(embMap.values())[0]
        idxs = list(idxsMap.values())[0]
        oriEmb = self.cacheData.get(name)
        if oriEmb is not None:
            emb = np.mean([emb, oriEmb], axis=0)
            print("recordFace ", len(emb), len(oriEmb))
        self.cacheData[name] = emb
        return True, idxs

    def saveCache(self):
        with open(self.cachePath, 'wb') as file_holder:
            pickle.dump(self.cacheData, file_holder)
        print("cacheData size ", len(self.cacheData))

    @staticmethod
    def dis(A, B):
        num = np.dot(A, B)  # 若为行向量则 A * B.T
        denom = np.linalg.norm(A) * np.linalg.norm(B)
        cos = num / denom  # 余弦值
        sim = 0.5 + 0.5 * cos  # 归一化
        return -sim

    def modelPredict(self, imgMap, isNeedTailOnlyFace=True):
        embMap = {}
        idxsMap = {}
        imgs = []
        keys = []
        for key, img in imgMap.items():
            if isNeedTailOnlyFace:
                status, idxs = self.getFaceRectangle(img)
                if status:
                    img = img[idxs[0]:idxs[1], idxs[2]:idxs[3]]
                    idxsMap[key] = idxs
                else:
                    continue
            img = cv2.resize(img, (self.size, self.size))
            keys.append(key)
            imgs.append(img)

        # 将图片数据与标签转换成数组
        imgs = np.array(imgs)
        # 参数：图片数据的总数，图片的高、宽、通道
        imgs = imgs.reshape(imgs.shape[0], self.size, self.size, 3)
        # 将数据转换成小于1的数
        imgs = imgs.astype('float32') / 255.0
        batch_size = 100
        num_batch = len(keys) // batch_size + 1 if len(keys) % batch_size > 0 else 0
        for i in range(num_batch):
            batch_imgs = imgs[i * batch_size: (i + 1) * batch_size]
            embeedings = self.sess.run(self.vgg.prob, feed_dict={self.input: batch_imgs})  # 提取最后一层的特征
            for idx, emb in enumerate(embeedings):
                embMap[keys[i * batch_size + idx]] = emb
        return embMap, idxsMap

    def stop(self):
        self.sess.close()

    def start(self):
        # 直接用python开启摄像头来工作
        pass

    def getPaddingSize(self, img):
        h, w, _ = img.shape
        top, bottom, left, right = (0, 0, 0, 0)
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

    def readData(self, path):
        imgs = []
        labs = []
        try:
            for filename in os.listdir(path):
                if filename.endswith('.jpg'):
                    filename = path + '/' + filename
                    img = cv2.imread(filename)
                    # top, bottom, left, right = self.getPaddingSize(img)
                    # # 将图片放大， 扩充图片边缘部分
                    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
                    imgs.append(img)
                    labs.append(path.split("/")[-1])
                else:
                    self.readData(path + "/" + filename)
        except Exception as e:
            print(e)
        return imgs, labs

    def createCacheData(self):
        imgs, labs = self.readData(self.facePath)
        imgMap = {}
        for idx, img in enumerate(imgs):
            imgMap[labs[idx]] = img
        cache, _ = self.modelPredict(imgMap)
        return cache


    def getFaceRectangle(self, img):
        # 获取头像所在矩形坐标,上下左右
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dets = self.detector(gray_image, 1)
        if not len(dets):
            return False, ""
        for i, d in enumerate(dets):
            x1 = d.top() if d.top() > 0 else 0
            x2 = d.bottom() if d.bottom() > 0 else 0
            y1 = d.left() if d.left() > 0 else 0
            y2 = d.right() if d.right() > 0 else 0
            return True, [x1,x2,y1,y2]

    def base64Img2cvImg(self, base64Img):
        imgData = base64.b64decode(base64Img)
        nparr = np.fromstring(imgData, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

if __name__ == '__main__':
    def runCam():
        cam = cv2.VideoCapture(0)
        while True:
            _, img = cam.read()
            break
    runCam()

