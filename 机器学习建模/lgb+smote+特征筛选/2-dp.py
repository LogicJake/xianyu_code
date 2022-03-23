import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体

warnings.filterwarnings('ignore')

os.makedirs('fig', exist_ok=True)

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

# 合并数据，方便预处理
df_feature = df_train.append(df_test).reset_index(drop=True)
del df_feature['Mobile_Tag']
del df_feature['ID']

# 异常值处理
# Client_Income 应该是数值特征，但部分值为 $，将其替换为0
df_feature['Client_Income'] = df_feature['Client_Income'].replace(
    '$', 0).astype(float)
# Loan_Annuity 应该是数值特征，但部分值为 #VALUE!，将其替换为0
df_feature['Loan_Annuity'] = df_feature['Loan_Annuity'].replace(
    '$', 0).replace('#VALUE!', 0).astype(float)
# Age_Days, Employed_Days, Registration_Days 应该是数值特征，但部分值为 #，将其替换为0
df_feature['Age_Days'] = df_feature['Age_Days'].replace('x', 0).astype(float)
df_feature['Employed_Days'] = df_feature['Employed_Days'].replace(
    'x', 0).astype(float)
df_feature['Registration_Days'] = df_feature['Registration_Days'].replace(
    'x', 0).astype(float)
df_feature['ID_Days'] = df_feature['ID_Days'].replace('x', 0).astype(float)
df_feature['Credit_Amount'] = df_feature['Credit_Amount'].replace(
    '$', 0).astype(float)
df_feature['Population_Region_Relative'] = df_feature[
    'Population_Region_Relative'].replace('@', 0).replace('#', 0).astype(float)
df_feature['Score_Source_3'] = df_feature['Score_Source_3'].replace(
    '&', 0).astype(float)

# 缺失值处理
# 数值特征缺失值用均值填充
f_list = []
for f in df_feature.select_dtypes(exclude='object').columns:
    if df_feature[f].isna().sum() != 0:
        # 标签列不处理
        if f in ['Default']:
            continue
        df_feature[f] = df_feature[f].fillna(df_feature[f].mean())
        f_list.append(f)
print('处理数值特征缺失值', f_list)

# 类别特征缺失值用众数填充
f_list = []
for f in df_feature.select_dtypes(include='object').columns:
    if df_feature[f].isna().sum() != 0:

        # 众数可能有多个，默认取第一个
        df_feature[f] = df_feature[f].fillna(df_feature[f].mode()[0])
        f_list.append(f)
print('处理类别特征缺失值', f_list)

# 数值特征归一化
f_list = []
for f in df_feature.select_dtypes(exclude='object').columns:
    # id列和标签列不处理
    if f in ['Default', 'ID', 'is_train']:
        continue

    df_feature[f] = (df_feature[f] - df_feature[f].min()) / (
        df_feature[f].max() - df_feature[f].min())

    f_list.append(f)

# 绘制散点图
for f in f_list:
    plt.cla()
    values = df_feature[f].unique()
    plt.scatter(range(len(values)), values)
    plt.title(f'{f}归一化散点图')
    plt.savefig(os.path.join('fig', f'{f}归一化散点图.png'))

# 类别特征 labelencoder
for f in df_feature.select_dtypes(include='object').columns:
    le = preprocessing.LabelEncoder()
    df_feature[f] = le.fit_transform(df_feature[f].astype('str'))

df_feature.to_csv('data1.csv', index=False)
