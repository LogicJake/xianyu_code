import json
import os
import time

import pandas as pd
import requests
from tqdm import tqdm

# 根据编号去具体页面获取正文内容
# api例子：https://api.vip.miui.com/api/community/post/feedbackDetail?postId=3629521
# 如果程序中断，出现“出现问题，提前退出”，不用担心，因为请求量很大，每个帖子单独请求一次，量大之后可能会被网站发现禁止
# 之前已经获取的数据是自动保存的，不用担心白跑
# 这时候可以等待一会重新运行，或者把 time.sleep 放大一点（同理，如果嫌他跑得慢，可以调小一点）
# 如果重新运行还是不行，截图找我
# PS：我运行的时候没遇到反爬，可能是我想多了


def get_content(id):
    url = "https://api.vip.miui.com/api/community/post/feedbackDetail"

    querystring = {"postId": id}

    headers = {
        'user-agent':
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/99.0.4844.74 Safari/537.36',
    }

    response = requests.get(url, headers=headers, params=querystring)

    resp = response.json()

    text = []
    title = resp['entity']['proposalTitle']
    text.append(title)

    textContent = resp['entity']['textContent']

    # 存在textContent不是字典，这时候可以跳过，内容就是title
    try:
        textContent = json.loads(textContent)

        for tc in textContent:
            text.append(tc['typeName'])
            text.append(tc['description'])
    except Exception:
        pass

    text = '\n'.join(text)
    return text


df = pd.read_csv('提案.csv', encoding='utf_8_sig')
# 初始化内容列
if '内容' not in df.columns:
    df['内容'] = ""
# 空白全部设置为空字符串，方便下面判断
df['内容'] = df['内容'].fillna("")

# 遍历每一行
for i, (id, text) in tqdm(enumerate(df[['编号', '内容']].values),
                          total=df.shape[0]):
    try:
        if len(text) < 1:
            text = get_content(id)
            # 填充到对应的内容位置
            df.loc[i, '内容'] = text
            time.sleep(0.1)
    except Exception as e:
        print('出现问题，提前退出', str(e), id)
        break

os.remove('提案.csv')
df.to_csv('提案.csv', index=False, encoding='utf_8_sig')
