import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体

warnings.filterwarnings('ignore')

df_train = pd.read_csv('data/train_smo.csv')
df_valid = pd.read_csv('data/test_smo.csv')

feature_names = [f for f in df_train.columns if f not in ['Default']]
target = 'Default'

print(df_train.shape, df_valid.shape)

model = xgb.XGBClassifier(random_state=2020,
                          learning_rate=0.04,
                          max_depth=10,
                          n_estimators=700,
                          reg_alpha=0,
                          reg_lambda=3)

model.fit(df_train[feature_names], df_train[target])
df_valid['prob'] = model.predict_proba(df_valid[feature_names])[:, 1]
df_train['prob'] = model.predict_proba(df_train[feature_names])[:, 1]
df_valid['label'] = model.predict(df_valid[feature_names])

df_valid['prob'].to_csv('test_xgb_predict.csv', index=False)
df_train['prob'].to_csv('train_xgb_predict.csv', index=False)

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
plt.title('xgboost ROC Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join('fig', 'xgboost roc曲线图.png'))
