from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
import jieba


# 文本特征提取
def countvec():
    # 实例化count
    count = CountVectorizer()

    data = count.fit_transform(["life is short,i like life","life is too long,i like my lovers","my daring"])

    print(count.get_feature_names())

    # 默认sparse矩阵
    print(data)
    print("#####################")
    # 利用toarray()进行sparse矩阵转换array数组
    print(data.toarray())
    return None


# 字典特征提取
def dictVe():

    # 进行实例化,字典向量化
    dict = DictVectorizer(sparse=False)
    """
     (0, 1)	1.0
     (0, 3)	100.0
     (1, 0)	1.0
     (1, 3)	60.0
     (2, 2)	1.0
     (2, 3)	30.0
    """

    data = dict.fit_transform([{'city': '北京','temperature':100},{'city': '上海','temperature':60}
                                ,{'city': '深圳','temperature':30}])# 返回sparse矩阵
    print(dict.get_feature_names())# 获取特征名称
    print(data)
    print(data.shape)
    return None


# 分词函数
def cutword():
    # 将句子进行jieba处理
    content1 = jieba.cut("今天很残酷，明天更残酷，后天很美好，但绝对大部分是死在明天晚上，所以每个人不要放弃今天。")
    content2 = jieba.cut("我们看到的从很远星系来的光是在几百万年之前发出的，这样当我们看到宇宙时，我们是在看它的过去")
    content3 = jieba.cut("如果只用一种方式了解某样事物，你就不会真正了解它。了解事物真正含义的秘密取决于如何将其与我们所了解的事物相联系。")
    # cccc = list(content1)
    # 将这三个转换成列表，编程以空格隔开的字符串
    c1 = ' '.join(list(content1))
    c2 = ' '.join(list(content2))
    c3 = ' '.join(list(content3))

    return c1,c2,c3


# 中文特征提取
def chineseCountVec():

    # 实例化count
    count = CountVectorizer(stop_words=['不会', '不要'])
    # count = CountVectorizer()
    # 定义一个分词函数
    c1,c2,c3 = cutword()

    data = count.fit_transform([c1,c2,c3])

    print(count.get_feature_names())
    print(data.toarray())


# Tf-idf文本特征提取
def chineseTfidf():

    # 实例化
    tfidf = TfidfVectorizer()

    # 定义一个分词函数
    c1,c2,c3 = cutword()

    data = tfidf.fit_transform([c1,c2,c3])

    print(tfidf.get_feature_names())
    print(data.toarray())


if __name__ == '__main__':

    # dictVe()
    # countvec()
    chineseCountVec()
    # chineseTfidf()