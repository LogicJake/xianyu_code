# 获取up主相关信息
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

if os.path.exists('data2.csv'):
    df = pd.read_csv('data2.csv', encoding='utf_8_sig')
else:
    df = pd.read_csv('data1.csv', encoding='utf_8_sig')

feats = ['level', 'fans', 'archive_count']
for f in feats:
    if f not in df:
        df[f] = 0

for i, mid in tqdm(enumerate(df['mid'].values), total=df.shape[0]):
    if df.loc[i, 'archive_count'] != 0:
        continue

    try:
        url = f'https://api.bilibili.com/x/web-interface/card?mid={mid}'

        rep = requests.get(
            url,
            headers={
                'user-agent':
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36',
            }).json()

        df.loc[i, 'level'] = rep['data']['card']['level_info']['current_level']
        df.loc[i, 'fans'] = rep['data']['card']['fans']
        df.loc[i, 'archive_count'] = rep['data']['archive_count']

    except Exception as e:
        print(i, mid, e)
        break

    time.sleep(0.5)

if os.path.exists('data2.csv'):
    os.remove('data2.csv')

df.to_csv('data2.csv', encoding='utf_8_sig', index=False)
