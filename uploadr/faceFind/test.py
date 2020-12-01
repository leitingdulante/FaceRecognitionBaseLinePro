import sys
import os
pro_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(pro_dir)

# import tensorflow as tf
#
# w1 = tf.Variable([[1,2]])
# w2 = tf.Variable([[3,4]])
#
# res = tf.matmul(w1, [[2],[1]])
#
# grads = tf.gradients(res,[w1])
#
# with tf.Session() as sess:
#     tf.global_variables_initializer().run()
#     print(sess.run(res))
#     print(sess.run(grads))

from sklearn.preprocessing import LabelBinarizer

def prepareData():
    lbe = LabelBinarizer()
    print(lbe.fit_transform(["jdod","diod","mmm","dd","ee", "ee", "ee"]))


prepareData()