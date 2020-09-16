
import tensorflow as tf

# import keras
# from keras.datasets import mnist

from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets

mnist =input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist.train.labels
# mnist.train.labels.shape
# mnist.train.images[0]
# mnist.train.images[0].shape


# mnist = tf.keras.datasets.mnist # 包含了很多数据集，第一次使用需要下载
#
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# print(X_train.shape) # out: (60000, 28, 28)
# print(y_train.shape) # out: (60000,)





