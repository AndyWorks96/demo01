import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 实现一个加法运算
con_a = tf.constant(3.0,name="con1")
con_b = tf.constant(4.0)

sum_ = tf.add(con_a, con_b)
print(con_a)
print(tf.get_default_graph())
with tf.Session() as sess:

    # 在会话当中序列化图到events文件
    file_writer=tf.summary.FileWriter("./tmp/summary", graph=sess.graph)