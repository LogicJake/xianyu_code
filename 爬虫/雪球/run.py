import time

import pandas as pd
import requests

header = {
    'User-Agent':
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
}

mp = input('请输入最大页数（最小值为1， 最大值为56）：')
mp = int(mp)

records = {}

name_map = {
    '股票代码': 'symbol',
    '股票名称': 'name',
    '当前价': 'current',
    '涨跌额': 'chg',
    '涨跌幅': 'percent',
    '年初至今': 'current_year_percent',
    '成交量': 'volume',
    '成交额': 'amount',
    '换手率': 'turnover_rate',
    '市盈率(TTM)': 'pe_ttm',
    '股息率': 'dividend_yield',
    '市值': 'market_capital',
}

for name in name_map:
    records[name] = []

for p in range(1, mp + 1):
    url = f'https://xueqiu.com/service/v5/stock/screener/quote/list?page={p}&size=30&order=desc&order_by=amount&exchange=CN&market=CN&type=sha'

    rep = requests.get(url, headers=header)
    rep = rep.json()
    data = rep['data']['list']

    for d in data:
        for name in name_map:
            key = name_map[name]

            records[name].append(d[key])

    time.sleep(0.5)

df = pd.DataFrame(records)
df.to_csv(f'记录1-{mp}.csv', index=False)
