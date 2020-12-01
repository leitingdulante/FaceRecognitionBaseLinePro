import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(pro_dir)

import pickle
import scipy
from uploadr.faceFind.faceUtil import *
from uploadr.faceFind import vgg19
readData(my_faces_path)
readData(other_faces_path)
labs, lbe = processLabel(LABEL_SIZE)
# 将图片数据与标签转换成数组
imgs = np.array(imgs)
# 参数：图片数据的总数，图片的高、宽、通道
imgs = imgs.reshape(imgs.shape[0], size, size, 3)
# 将数据转换成小于1的数
imgs = imgs.astype('float32')/255.0
# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs, test_size=0.1, random_state=random.randint(0,100))
print('train size:%s, test size:%s' % (len(train_x), len(test_x)))
# 图片块，每次取128张图片
batch_size = 256
num_batch = len(train_x) // batch_size

def cnnTrain():
    # 进行网络结构改造
    # 1.预训练的网络就是一个特征提取器，最终输出一个特定维度比如32的向量，这个要是最后的softmax之前的固定维度的，不随着训练样本变化的向量，也可以是embeeding层的
    # 2.新来一个用户，需要录入到系统，简单点就是直接用模型进行特征提取，提取完成之后，暂存，复杂点就是对该图片也进行增量训练，不过需要预留出softmax的位置
    # 3.识别的时候就是检测到人脸情况下，用模型进行特征提取，提取后向量和系统中的向量求余弦相似度最大值且超过一定阈值，然后用户识别后进行相应提醒
    # Phil_Donahue
    out, embeedingTensor = cnnLayer(LABEL_SIZE)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y_))

    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)
    # 比较标签是否相等，再求的所有数的平均值，tf.cast(强制转换类型)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(out, 1), tf.argmax(y_, 1)), tf.float32))
    # 将loss与accuracy保存以供tensorboard使用
    tf.summary.scalar('loss', cross_entropy)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    # 数据保存器的初始化
    saver = tf.train.Saver()

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('./tmp', graph=tf.get_default_graph())

        for n in range(60):
             # 每次取128(batch_size)张图片
            for i in range(num_batch):
                batch_x = train_x[i*batch_size : (i+1)*batch_size]
                batch_y = train_y[i*batch_size : (i+1)*batch_size]
                # 开始训练数据，同时训练三个变量，返回三个数据
                embeedings, _,loss,summary = sess.run([embeedingTensor, train_step, cross_entropy, merged_summary_op],
                                           feed_dict={x:batch_x,y_:batch_y, keep_prob_5:0.5,keep_prob_75:0.75})
                summary_writer.add_summary(summary, n*num_batch+i)
                print(tf.reduce_mean(embeedings, 0).eval())
                # 打印损失
                print(n*num_batch+i, loss)

                if (n*num_batch+i) % 100 == 0:
                    # 获取测试数据的准确率
                    acc = accuracy.eval({x:test_x, y_:test_y, keep_prob_5:1.0, keep_prob_75:1.0})
                    print(n*num_batch+i, acc, '-------------------------')
                    # 准确率大于0.98时保存并退出
                    if acc > 0.1 and n > 2:
                        saver.save(sess, './train_faces.model', global_step=n*num_batch+i)
                        sys.exit(0)
        saver.save(sess, './train_faces.model')
        print('accuracy less 0.98, exited!')
        sys.exit(0)

cacheDb = {}
def saveCache(key, vector):
    cacheDb[key] = vector

def cnnCache():
    # 进行网络结构改造
    # 1.预训练的网络就是一个特征提取器，最终输出一个特定维度比如32的向量，这个要是最后的softmax之前的固定维度的，不随着训练样本变化的向量，也可以是embeeding层的
    # 2.新来一个用户，需要录入到系统，简单点就是直接用模型进行特征提取，提取完成之后，暂存，复杂点就是对该图片也进行增量训练，不过需要预留出softmax的位置
    # 3.识别的时候就是检测到人脸情况下，用模型进行特征提取，提取后向量和系统中的向量求余弦相似度最大值且超过一定阈值，然后用户识别后进行相应提醒
    # Phil_Donahue
    # out, embeedingTensor = cnnLayer(LABEL_SIZE)
    # saver = tf.train.Saver()
    # sess = tf.Session()
    # saver.restore(sess, tf.train.latest_checkpoint('.'))
    #
    # embeedings = sess.run(embeedingTensor, feed_dict={x: imgs, keep_prob_5: 1.0, keep_prob_75: 1.0})
    # oriLabls = lbe.inverse_transform(labs)
    # for idx, embeeding in enumerate(embeedings):
    #     saveCache(oriLabls[idx], embeeding)
    #
    # oriTestLals = lbe.inverse_transform(test_y)
    # for idx,test_x_one in enumerate(test_x):
    #     maxSimKey, maxSimValue = predict(sess, embeedingTensor, test_x_one)
    #     print(oriTestLals[idx], maxSimKey, maxSimValue)
    # sess.close()

    sess = tf.Session()
    oriLabls = lbe.inverse_transform(labs)
    vgg = vgg19.Vgg19()
    vgg.build(x)

    batch_size = 100
    num_batch = len(imgs) // batch_size
    for i in range(num_batch):
        print(i, "end")
        batch_imgs = imgs[i * batch_size: (i + 1) * batch_size]
        embeedings = sess.run(vgg.fc7, feed_dict={x: batch_imgs})  # 提取fc7层的特征
        for idx, feature in enumerate(embeedings):
            saveCache(oriLabls[i * batch_size + idx], feature)

    with open("./featureCache", 'wb') as file_holder:
        pickle.dump(cacheDb, file_holder)
    # global cacheDb
    # with open("./featureCacheProb", 'rb') as file_holder:
    #     cacheDb = pickle.load(file_holder)

    print("cache end----------")
    oriTestLals = lbe.inverse_transform(test_y)
    # 统计正确率
    rightCount = 0
    for idx,test_x_one in enumerate(test_x):
        simKey, simValue = predict(sess, vgg.fc7, test_x_one)
        print(oriTestLals[idx], simKey, simValue)
        if oriTestLals[idx] == simKey:
            rightCount += 1
    print(len(test_x), rightCount, rightCount/len(test_x))
    sess.close()

def dis(A, B):
    num = np.dot(A, B)  # 若为行向量则 A * B.T
    denom = np.linalg.norm(A) * np.linalg.norm(B)
    cos = num / denom  # 余弦值
    sim = 0.5 + 0.5 * cos  # 归一化
    return -sim

def predict(sess, embeedingTensor, img, threshold = -1):
    # embeedings = sess.run(embeedingTensor, feed_dict={x: [img], keep_prob_5: 1.0, keep_prob_75: 1.0})
    embeedings = sess.run(embeedingTensor, feed_dict={x: [img]})
    embeeding = embeedings[0]
    # embeeding /= scipy.linalg.norm(embeeding)  # 特征归一化
    simValue = 10000
    simKey = None
    for key, vector in cacheDb.items():
        newDis = dis(embeeding, vector)
        if newDis < simValue:
            isSet = True
            if threshold != -1 and newDis > threshold:
                isSet = False
            if isSet:
                simValue = newDis
                simKey = key
    return simKey, simValue

# cnnTrain()

cnnCache()
