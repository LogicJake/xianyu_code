import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, roc_auc_score,
                             roc_curve)

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体

warnings.filterwarnings('ignore')
df_train = pd.read_csv('data/train_smo.csv')
df_valid = pd.read_csv('data/test_smo.csv')

df_train = df_train[['Default']]
df_valid = df_valid[['Default']]

feature_names = ['lgb', 'xgb', 'rf']
for name in feature_names:
    df1 = pd.read_csv('train_{}_predict.csv'.format(name))
    df_train[name] = df1['prob']

    df2 = pd.read_csv('test_{}_predict.csv'.format(name))
    df_valid[name] = df2['prob']

print(df_train.head())
print(df_valid.head())

model = LogisticRegression(random_state=2022, max_iter=10)
model.fit(df_train[feature_names], df_train['Default'])

df_valid['prob'] = model.predict_proba(df_valid[feature_names])[:, 1]
df_valid['label'] = model.predict(df_valid[feature_names])

acc = accuracy_score(df_valid['Default'], df_valid['label'])
auc = roc_auc_score(df_valid['Default'], df_valid['prob'])
cm = confusion_matrix(df_valid['Default'], df_valid['label'])

print(acc, auc)
print(cm)

fpr, tpr, thersholds = roc_curve(df_valid['Default'], df_valid['prob'])
plt.plot(fpr, tpr, label='ROC (area = {0:.2f})'.format(auc), lw=2)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('stacking ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join('fig', 'stacking roc曲线图.png'))
