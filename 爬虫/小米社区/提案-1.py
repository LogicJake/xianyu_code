import time

import pandas as pd
import requests
from tqdm import tqdm

# 提案要分两个步骤来做，首先在列表页获取编号，标签和时间

after = ""
records = {'编号': [], '标签': [], '时间': []}

for _ in tqdm(range(10)):
    url = "https://prod.api.xiaomi.cn/community/board/search/announce/list"

    querystring = {
        "after": after,  # after代表这段的数据获取过了，获取后面的数据
        "limit": "100",  # 返回的数据条数
        "boardId":
        "23377339",  # 小米11的圈子id，可以从圈子url得到：https://www.xiaomi.cn/board/23377339
        "profileType": "6",  # 6 代表提案栏目
    }

    headers = {
        'user-agent':
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36',
    }

    response = requests.get(url, headers=headers, params=querystring)

    # api链接返回的是一个json字典，可以点开这个链接看格式：https://prod.api.xiaomi.cn/community/board/search/announce/list?after=&limit=1&boardId=23377339&profileType=6&displayName=%E6%8F%90%E6%A1%88
    resp = response.json()

    # 获取当前数据的范围区间
    after = resp['entity']['after']
    # 所有贴子信息
    posts = resp['entity']['records']

    # 遍历每个帖子，post也是个字典
    for post in posts:
        # 提案编号
        id = post['id']

        # 提案里面标签都在board
        tags = []

        for t in post['boards']:
            tags.append(t['boardName'])

        # 拼接成字符串
        tags = ','.join(tags)
        # 创建时间
        create_time = post['createTime']
        # 原始数据是时间戳，转成普通时间
        timeStamp = float(create_time / 1000)
        timeArray = time.localtime(timeStamp)
        style_time = time.strftime("%Y-%m-%d %H:%M:%S", timeArray)

        records['编号'].append(id)
        records['时间'].append(style_time)
        records['标签'].append(tags)

    # 休眠1s，防止太快反爬，避免干扰网站正常运行
    time.sleep(1)

df = pd.DataFrame(records)
df.to_csv('提案.csv', index=False, encoding='utf_8_sig')
