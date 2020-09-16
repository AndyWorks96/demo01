import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 实现一个加法运算
with tf.variable_scope("my_scope"):
    con_a = tf.constant(3.0,name="con1")
    con_b = tf.constant(4.0,name="con1")
    sum_ = tf.add(con_a, con_b)
print(con_a)
print(con_b)
print(sum_)


# 特殊的创建张量op
# 1.必须手动初始化
# variable大小写错误
var = tf.Variable(tf.random_normal([2,3], mean=0.0, stddev=1.0), name="var_name")
init_var=tf.global_variables_initializer()
new_var=var.assign([[1,2,3],[4,5,6]])
var1=var.assign_add([[1,2,3],[4,5,6]])
print(var)
with tf.Session() as sess:
    sess.run(init_var)
    print(sess.run([var1,new_var]))


