import pandas as pd

df = pd.read_csv('data.csv')

if '链接' in df:
    del df['链接']

data1 = pd.DataFrame()
data2 = pd.DataFrame()

for _, g in df.groupby('所在区'):
    data1 = data1.append(g.iloc[:(g.shape[0] - 100)])
    data2 = data2.append(g.iloc[(g.shape[0] - 100):])

print(data1.shape, data2.shape)

data1.to_csv('house_bj.csv', index=False)
data2.to_csv('house_bj_new.csv', index=False)
