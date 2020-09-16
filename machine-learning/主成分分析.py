import pandas as pd
from sklearn.decomposition import PCA
# 导入四张表
prior = pd.read_csv("../data/instacart/order_products__prior.csv")
aisles = pd.read_csv("../data/instacart/aisles.csv")
products = pd.read_csv("../data/instacart/products.csv")
orders = pd.read_csv("../data/instacart/orders.csv")
# 合并四张表到一张表中
# on指定两张表共同拥有的键
mt = pd.merge(prior, products, on=['product_id','product_id'])
mt1 = pd.merge(mt, orders, on=['order_id','order_id'])
mt2 = pd.merge(mt1,aisles,on=['aisle_id','aisle_id'])

# print(mt2.shape)

# 进行交叉表交换，用户，跟商品类别的分组次数统计
user_aisle = pd.crosstab(mt2['user_id'],mt2['aisle'])
# [206209 rows x 134 columns]
# print(user_aisle)


# 主成分分析
pc = PCA(n_components=0.95)
data = pc.fit_transform(user_aisle)

#[206209 rows x 44 columns]
print(pd.DataFrame(data))