import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 实现一个加法运算
a=tf.constant(3.0)
b=tf.constant(4.0)

sum = tf.add(a,b)

# 图：打印出来，就是一个分配内存的地址
# more所有的张量、op、会话都在一张图当中
# print(sum)
print(tf.get_default_graph())

# 注意分号


# 创建图
g=tf.Graph()

# with g.as_default():
    # 在g图中定义了一个operation
    # c=tf.constant(30.0)
    # print(c.graph)

# 运行会话并打印设备信息
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                            log_device_placement=True)) as sess:
    # print(sess.run(sum))

    print(sess.run(sum))

#TODO
    # assert c.graph is g