import os
import warnings

import pandas as pd

warnings.filterwarnings('ignore')

os.makedirs('data', exist_ok=True)

df = pd.read_csv('raw_data/Train_Dataset.csv/Train_Dataset.csv')

df_train = df[:int(df.shape[0] * 0.9)]
df_valid = df[int(df.shape[0] * 0.9):]

df_train['is_train'] = 1
df_valid['is_train'] = 0

print('训练集数量', df_train.shape)
print('测试集数量', df_valid.shape)

df_train.to_csv(os.path.join('data', 'train.csv'), index=False)
df_valid.to_csv(os.path.join('data', 'test.csv'), index=False)
