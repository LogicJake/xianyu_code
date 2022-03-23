import warnings

import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

df = pd.read_csv('data.csv')

feature_names = [f for f in df.columns if f not in ['y']]

# 定义弹性网络
model = ElasticNet()

x = df[feature_names]
y = df['y']

# 不调参数的结果
model = ElasticNet()
mse = -cross_val_score(model, x, y, cv=5,
                       scoring='neg_mean_squared_error').mean()
print('调参前mse：', mse)


# 目标函数，5折交叉验证
def obj(alpha, l1_ratio):
    val = cross_val_score(ElasticNet(alpha=alpha, l1_ratio=l1_ratio),
                          x,
                          y,
                          scoring='neg_mean_squared_error',
                          cv=5).mean()
    return val


bo = BayesianOptimization(obj, {
    'alpha': [0, 1],
    'l1_ratio': [0, 1],
},
                          random_state=2022,
                          verbose=0)

# 开始搜索
bo.maximize()

params = bo.max['params']
alpha = params['alpha']
l1_ratio = params['l1_ratio']

# 调参后的结果
model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
mse = -cross_val_score(model, x, y, cv=5,
                       scoring='neg_mean_squared_error').mean()
print('调参后mse：', mse)
model.fit(x, y)

# l1和l2权重的换算：https://blog.csdn.net/qq_41386300/article/details/99115512
l1_weight = alpha * l1_ratio
l2_weight = alpha * (1 - l1_ratio) / 2
print('搜索的最优参数 l1_weight: {} l2_weight: {} 权重w: {}, 截距b: {}'.format(
    l1_weight, l2_weight, model.coef_, model.intercept_))

print('迭代日志')
for i, res in enumerate(bo.res):
    alpha = res['params']['alpha']
    l1_ratio = res['params']['l1_ratio']
    mse = -res['target']

    l1_weight = alpha * l1_ratio
    l2_weight = alpha * (1 - l1_ratio) / 2

    print("{}: l1_weight: {} l2_weight: {} mse: {}".format(
        i, l1_weight, l2_weight, mse))
