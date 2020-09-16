import tensorflow as tf
import numpy as np

# 实现一个加法运算
a = tf.constant(3.0)
b = tf.constant(4.0)

# 可以使用重载的运算 + --> 加法op操作
sum_ma = b + b
print(sum_ma)

sum_ = tf.add(a, b)

# 结合feed_dict使用
# 当不确定数据的形状，可以使用none
# [None, 3]
plt = tf.placeholder(tf.float32, [None, 3])

# print(sum)
# 会话,默认只能运行默认的图，不能运行其它的图（可以通过graph参数解决）
# 1、会话：运行图结构
# 2、会话掌握了资源，会话运行结束，资源释放，无法再去使用这些资源计算
# with : , close()
with tf.Session() as sess:
    # run你要运行的内容, 必须是一个op
    # 允许调用的时候去覆盖原来的值，运行时候提供数据
    print(sess.run([sum_, sum_ma, a, b, plt], feed_dict = {plt: [[1, 2, 3], [4, 5, 6]]}))
