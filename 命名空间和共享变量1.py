import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 实现一个加法运算
with tf.variable_scope("my_scope"):
    con_a = tf.constant(3.0,name="con1")
    con_b = tf.constant(4.0,name="con1")
    sum_ = tf.add(con_a, con_b)
# print(con_a)
# print(con_b)
# print(sum_)


# 特殊的创建张量op
# 1.必须手动初始化
# variable大小写错误
# with tf.variable_scope("my_scope01") as scope:
with tf.variable_scope("my_scope01",reuse=tf.AUTO_REUSE):
    var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0), name="var1")
    var_double = tf.Variable(tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="var1")
    # 使用get_variable 此时会报错误
    # scope.reuse_variables() 此处出现报错需在下边
    #  在需要使用共享变量的前面定义： scope.reuse_variables()
    var2 = tf.get_variable(initializer=tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="var2")
    # scope.reuse_variables()
    # initializer参数必须指定
    var2_double = tf.get_variable(initializer=tf.random_normal([2, 3], mean=0.0, stddev=1.0), name="var2")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # print(sess.run([var,var_double]))
    print(var)
    print(var_double)
    print(var2)
    print(var2_double)


