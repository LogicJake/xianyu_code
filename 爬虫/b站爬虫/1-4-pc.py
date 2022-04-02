# 获取视频互动数据，除了热评

import os
import time

import pandas as pd
import requests
from tqdm import tqdm

if os.path.exists('data.csv'):
    df = pd.read_csv('data.csv', encoding='utf_8_sig')
else:
    df = pd.read_csv('data3.csv', encoding='utf_8_sig')

for i in range(10):
    f = 'hot_comment{}'.format(i)
    if f not in df:
        df[f] = ""

for i, aid in tqdm(enumerate(df['id'].values), total=df.shape[0]):
    if df.loc[i, 'hot_comment1'] != "":
        continue

    try:
        url = f'https://api.bilibili.com/x/v2/reply?%20&jsonp=jsonp&pn=1&type=1&oid={aid}&sort=2'

        rep = requests.get(
            url,
            headers={
                'user-agent':
                'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36',
            }).json()

        for j in range(10):
            f = 'hot_comment{}'.format(j)
            df.loc[i, f] = rep['data']['replies'][j]['content']['message']

    except Exception as e:
        print(i, aid, e)
        break
    time.sleep(0.5)

if os.path.exists('data.csv'):
    os.remove('data.csv')

df.to_csv('data.csv', encoding='utf_8_sig', index=False)
