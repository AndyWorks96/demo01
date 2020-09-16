import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def mm():
    """
    对二维数组进行归一化处理
    :return:
    """
    data = pd.read_csv("../data/datingtest/dating.txt")
    print(data)
    print(data.shape)
    mm = MinMaxScaler(feature_range=(0, 1))

    data = mm.fit_transform(data[['milage','Liters','Consumtime']])

    print(data)
    print(data.shape)

    return None


def standScaler():

    """
    对二维数组进行标准化处理
    :return:
    """
    data = pd.read_csv("../data/datingtest/dating.txt")
    # 实例化 X'=(x-mean)/φ
    stand = StandardScaler()

    data = stand.fit_transform(data[['milage','Liters','Consumtime']])

    print(data)
    print(data.shape)

    return None

# 特征选择 低方差特征过滤
def varthreshold():
    """
    使用方差法进行指标过滤
    删除所有低方差特征
    :return:
    """
    fator = pd.read_csv("../data/datingtest/dating.txt")
    print(fator)

    # 使用VarianceThreshold,9列进行低方差过滤
    var = VarianceThreshold(threshold=0.0)

    data = var.fit_transform(fator.iloc[:, 1:4])
    print(data)
    return None


# 皮尔森相关系数
def pearSonr():
    """
    一些系数的相关计算
    :return:
    """
    factor = ['milage','Liters','Consumtime','target']
    data = pd.read_csv("../data/datingtest/dating.txt")
    ddd = data[factor[0]]
    print(ddd)
    # 循环获取两个指标
    for i in range(len(factor)):
        for j in range(i,len(factor)-1): # 0-11（1-10）
            print("指标%s和指标%s之间的相关性大小%f" %(
                factor[i],
                factor[j+1],
                pearsonr(data[factor[i]], data[factor[j+1]])[0] # 是一个元组，有两个系数，取第一个
            ))
    plt.scatter(data['milage'],data['Liters'])
    plt.show()
    return None


def pc_a():
    """
    主成分分析降维
    将数据分解为较低维数空间
    n_components:
        小数：表示保留百分之多少的信息
        整数：减少到多少特征
    PCA.fit_transform(X) X:numpy array格式的数据[n_samples,n_features]
    返回值：转换后指定维度的array

    """
    data = pd.read_csv("../data/datingtest/dating.txt")
    pca = PCA(n_components=1)# 整数代表减少到几个特征，一般选择小数
    # pca=PCA()
    # data = pca.fit_transform([[2,8,4,5],[6,3,0,8],[5,4,9,1]])
    data = pca.fit_transform(data[['milage','Liters']])# 对两个特征进行压缩
    print(data)
    return None


if __name__ == '__main__':

    # mm()
    # standScaler()
    # varthreshold()
    # pearSonr()
    pc_a()