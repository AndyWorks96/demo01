import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=(2, 2))
x = tf.constant([[1,1],[1,1]])
y = tf.matmul(x, x)

with tf.Session() as sess:
    #  print(sess.run(y)) ERROR此处x还没有赋值
    #  rand_array = np.random.rand(3, 3)
    # print(sess.run(y, feed_dict={x: rand_array}))
    print(sess.run(y))