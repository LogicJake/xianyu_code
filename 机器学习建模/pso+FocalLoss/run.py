import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from pso import PSO


def focal_loss(y, p, alpha, gamma):
    loss = -alpha * y * np.log(p) * (1 - p)**gamma - (1 - alpha) * (
        1 - y) * np.log(1 - p) * p**gamma
    loss = np.mean(loss)
    return loss


# 加载数据
df = pd.read_csv('数据.csv')
# 定义label列
label = 'y'
# 定义特征列
features = [f for f in df.columns if f != label]

X = df[features]
y = df[label]
# 划分训练集，测试集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=42)
# focal_loss 参数
gamma = 0.1
alpha = 0.1


def focal_loss_objective(y, p):
    # sigmoid 转为概率
    p = 1.0 / (1.0 + np.exp(-p))

    # focal_loss 一阶导
    grad = p * (1 - p) * (alpha * gamma * y * (1 - p)**gamma * np.log(p) /
                          (1 - p) - alpha * y *
                          (1 - p)**gamma / p - gamma * p**gamma * (1 - alpha) *
                          (1 - y) * np.log(1 - p) / p + p**gamma *
                          (1 - alpha) * (1 - y) / (1 - p))
    # focal_loss 二阶导
    hess = p * (1 - p) * (
        p * (1 - p) *
        (-alpha * gamma**2 * y * (1 - p)**gamma * np.log(p) /
         (1 - p)**2 + alpha * gamma * y * (1 - p)**gamma * np.log(p) /
         (1 - p)**2 + 2 * alpha * gamma * y * (1 - p)**gamma /
         (p * (1 - p)) + alpha * y *
         (1 - p)**gamma / p**2 - gamma**2 * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p**2 + 2 * gamma * p**gamma * (1 - alpha) *
         (1 - y) / (p * (1 - p)) + gamma * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p**2 + p**gamma * (1 - alpha) * (1 - y) /
         (1 - p)**2) - p *
        (alpha * gamma * y * (1 - p)**gamma * np.log(p) / (1 - p) - alpha * y *
         (1 - p)**gamma / p - gamma * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p + p**gamma * (1 - alpha) * (1 - y) /
         (1 - p)) + (1 - p) *
        (alpha * gamma * y * (1 - p)**gamma * np.log(p) / (1 - p) - alpha * y *
         (1 - p)**gamma / p - gamma * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p + p**gamma * (1 - alpha) * (1 - y) /
         (1 - p)))

    return grad, hess


# 未调参前训练
model = xgb.XGBClassifier(objective=focal_loss_objective,
                          use_label_encoder=False)
model.fit(X_train, y_train, eval_metric='logloss')

y_predict = model.predict_proba(X_test)[:, 1]
loss = focal_loss(y_test, y_predict, alpha=alpha, gamma=gamma)
auc = roc_auc_score(y_test, y_predict)
print('调参前Focal loss, AUC: {} {}'.format(loss, auc))

# PSO 搜索
# n_estimators, max_depth, learning_rate, min_child_weight, colsample_bytree, alpha, gamma的下界
low_boud = [50, 5, 0.001, 1, 0.1, 0, 0.1]
# n_estimators, max_depth, learning_rate, min_child_weight, colsample_bytree, alpha, gamma的上界
high_bound = [1000, 20, 0.1, 10, 1, 0.9, 0.9]
pso = PSO(10, 2, 7, low_boud, high_bound, X_train, X_test, y_train, y_test)
fitness_list = pso.search()

# pso收敛图

plt.cla()
plt.title('Fitness VS PSO rounds')
plt.xlabel('PSO rounds')
plt.ylabel('Fitness')

plt.plot(range(1, pso.rounds + 1), fitness_list)
plt.savefig('pso.png')

# 使用调参后参数训练
print('最优参数为: ', pso.g_best)
n_estimators = int(pso.g_best[0])
max_depth = int(pso.g_best[1])
learning_rate = pso.g_best[2]
min_child_weight = pso.g_best[3]
colsample_bytree = pso.g_best[4]
# pso 2个参数
alpha = pso.g_best[5]
gamma = pso.g_best[6]


def focal_loss_objective(y, p):
    # sigmoid 转为概率
    p = 1.0 / (1.0 + np.exp(-p))

    # focal_loss 一阶导
    grad = p * (1 - p) * (alpha * gamma * y * (1 - p)**gamma * np.log(p) /
                          (1 - p) - alpha * y *
                          (1 - p)**gamma / p - gamma * p**gamma * (1 - alpha) *
                          (1 - y) * np.log(1 - p) / p + p**gamma *
                          (1 - alpha) * (1 - y) / (1 - p))
    # focal_loss 二阶导
    hess = p * (1 - p) * (
        p * (1 - p) *
        (-alpha * gamma**2 * y * (1 - p)**gamma * np.log(p) /
         (1 - p)**2 + alpha * gamma * y * (1 - p)**gamma * np.log(p) /
         (1 - p)**2 + 2 * alpha * gamma * y * (1 - p)**gamma /
         (p * (1 - p)) + alpha * y *
         (1 - p)**gamma / p**2 - gamma**2 * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p**2 + 2 * gamma * p**gamma * (1 - alpha) *
         (1 - y) / (p * (1 - p)) + gamma * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p**2 + p**gamma * (1 - alpha) * (1 - y) /
         (1 - p)**2) - p *
        (alpha * gamma * y * (1 - p)**gamma * np.log(p) / (1 - p) - alpha * y *
         (1 - p)**gamma / p - gamma * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p + p**gamma * (1 - alpha) * (1 - y) /
         (1 - p)) + (1 - p) *
        (alpha * gamma * y * (1 - p)**gamma * np.log(p) / (1 - p) - alpha * y *
         (1 - p)**gamma / p - gamma * p**gamma * (1 - alpha) *
         (1 - y) * np.log(1 - p) / p + p**gamma * (1 - alpha) * (1 - y) /
         (1 - p)))

    return grad, hess


model = xgb.XGBClassifier(objective=focal_loss_objective,
                          use_label_encoder=False,
                          n_estimators=n_estimators,
                          max_depth=max_depth,
                          learning_rate=learning_rate,
                          min_child_weight=min_child_weight,
                          colsample_bytree=colsample_bytree)
model.fit(X_train, y_train, eval_metric='logloss')

y_predict = model.predict_proba(X_test)[:, 1]
loss = focal_loss(y_test, y_predict, alpha=alpha, gamma=gamma)
auc = roc_auc_score(y_test, y_predict)
print('调参后Focal loss, AUC: {} {}'.format(loss, auc))

# roc 曲线图
plt.cla()
fpr, tpr, thersholds = roc_curve(y_test, y_predict)
plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(auc), lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc.png')
