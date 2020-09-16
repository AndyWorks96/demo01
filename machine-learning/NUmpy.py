"""
    numpy的详细使用
    author：lixing
    2020/08/20
"""
import numpy as np
import matplotlib.pyplot as plt
# 创建数组
np1 = np.array([1,2,3,4,5])
np2 = np.array(range(5))
np3 = np.arange(5)
np4 = np.arange(1,10,3) # 第三个参数为步长
np5 = np.linspace(1,10,11) # 第三个参数为生成样本数
np6 = np.linspace(1,10,10,endpoint=True) #true包含stop否则不包含

""""
    创建一个等比数列 指定开始值，结束值 ，元素个数，不指定对数底数，默认为10
"""
np7= np.logspace(0,3,4)
np8 = np.zeros(shape=(2,3))
np9 = np.ones(shape=(2,3))
# TypeError: 'tuple' object cannot be interpreted as an integer
# 类型错误：元组对象不能被解释为一个整型
np10 = np.identity(3,dtype=bool)
np11 = np.empty((3,3))
# help(np.empty)# 查看说明文档
# 汉明窗（Hamming window）形式上是一个加权的余弦函数。NumPy中的hamming函数返回汉明窗。
np12 = np.hamming(6)
# plt.plot(np12)
# plt.show()
# print(np11)

"""
    在python中计算一个多维数组的任意百分比分位数，此处的百分位是从小到大排列
"""
a = np.array([[1,3,5],[2,4,6]])
b = np.percentile(a,50,axis=0)
c= np.percentile(a,50,axis=1,keepdims=True)# keepdims维度保持不变
print(b)
print(c)
print(c.shape)