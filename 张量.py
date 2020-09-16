import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tensor1 = tf.constant(4.0)
tensor2 = tf.constant([1, 2, 3, 4])
x = tf.constant([[1,1],[1,1],[1,1]])
linear_squares = tf.constant([[4], [9], [16], [25]], dtype=tf.int32)

print(tensor1.shape)
print(tensor2.shape)
print(x.shape)
print(linear_squares.shape)
# 0维：()   1维：(10, )   2维：(3, 4)   3维：(3, 4, 5)
# tf.fill(dims,value,name=None)
y=tf.random_normal(shape=(3,4),mean=0,stddev=1.0,dtype=tf.float32,seed=1,name="random11")
with tf.Session() as sess:

    print(sess.run(y))
    print(y)
    cast=tf.cast(x,dtype=tf.float32,name=None)
    print(sess.run(cast))