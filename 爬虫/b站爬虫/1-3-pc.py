# 获取视频互动数据，除了热评

import os
import time

import pandas as pd
import requests
from tqdm import tqdm

if os.path.exists('data3.csv'):
    df = pd.read_csv('data3.csv', encoding='utf_8_sig')
else:
    df = pd.read_csv('data2.csv', encoding='utf_8_sig')

feats = ['like', 'coin', 'favorite', 'share', 'view', 'danmaku', 'reply']
for f in feats:
    if f not in df:
        df[f] = 0

for i, aid in tqdm(enumerate(df['id'].values), total=df.shape[0]):
    if df.loc[i, 'view'] != 0:
        continue

    try:
        url = f'https://api.bilibili.com/x/web-interface/archive/stat?aid={aid}'

        rep = requests.get(
            url,
            headers={
                'user-agent':
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36',
            }).json()

        for f in feats:
            df.loc[i, f] = rep['data'][f]
    except Exception as e:
        print(i, aid, e)
        break
    time.sleep(0.5)

if os.path.exists('data3.csv'):
    os.remove('data3.csv')

df.to_csv('data3.csv', encoding='utf_8_sig', index=False)
