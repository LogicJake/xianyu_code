import pandas as pd
from imblearn.over_sampling import SMOTE
import os

df = pd.read_csv('data.csv')

df_train = df[df['is_train'] == 1]
df_valid = df[df['is_train'] == 0]

print(df_train['Default'].value_counts())

feature_names = [f for f in df.columns if f not in ['Default', 'is_train']]

smo = SMOTE(random_state=42)
X_smo, y_smo = smo.fit_resample(df_train[feature_names], df_train['Default'])

X_smo['Default'] = y_smo
X_smo['is_train'] = 1

print(X_smo['Default'].value_counts())

df_train.to_csv(os.path.join('data', 'train_smo.csv'), index=False)
df_valid.to_csv(os.path.join('data', 'test_smo.csv'), index=False)
