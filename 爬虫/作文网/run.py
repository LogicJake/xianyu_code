import random
import re
import urllib.request
from bs4 import BeautifulSoup

import os

current_path = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(current_path, 'spider'), exist_ok=True)

agents = [
    "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36"
]

# 随机选取一个header防止反爬
headers = {"User-Agent": random.choice(agents)}

main_url = 'https://www.diyifanwen.com/zuowen/gaozhongzuowen/gkmfzw/18211540432881183.htm'

req = urllib.request.Request(url=main_url, headers=headers)
rep = urllib.request.urlopen(req)
soup = BeautifulSoup(rep.read(), 'lxml')

papers = soup.select('ul.data a')

for paper in papers:
    name = paper['title'].replace('?', '？')
    url = paper['href']

    # 使用正则提取网址
    url = re.findall(r'//(.*)', url)[0]

    rep = urllib.request.urlopen(url='https://' + url)
    soup = BeautifulSoup(rep.read(), 'lxml')

    contents = []
    for content in soup.select('div#ArtContent > p'):
        contents.append(content.get_text() + '\n')

    with open(os.path.join(current_path, 'spider', name + '.txt'),
              'w',
              encoding='utf_8_sig') as f:
        f.writelines(contents)
