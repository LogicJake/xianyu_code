import os

import numpy as np
import pandas as pd
import requests
from lxml import etree
from tqdm import tqdm

if os.path.exists('cache.npy'):
    cache = np.load('cache.npy', allow_pickle=True)
    cache = cache.item()
else:
    cache = {}

df = pd.read_csv('data1.csv')

for url in tqdm(df['链接'].values):
    if url in cache:
        continue
    else:
        try:
            rep = requests.get(url)
            content = rep.text

            root = etree.HTML(content)
            dt = root.xpath(
                '//*[@id="introduction"]/div/div/div[1]/div[2]/ul/li[12]/text()'
            )

            if len(dt) == 0 or dt == '暂无数据':
                dt = None
            else:
                dt = dt[0]
                dt = dt + '电梯'

            cache[url] = dt
        except Exception as e:
            print(url, e)
            break

np.save('cache.npy', cache)

df['有无电梯'] = df['链接'].map(cache)
df = df.dropna()
df.to_csv('data.csv', index=False)
