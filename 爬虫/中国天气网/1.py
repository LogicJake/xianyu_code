import re

import pandas as pd
import requests
from lxml import etree

# 先访问北京页面取到所有的省份链接
rep = requests.get('http://www.weather.com.cn/textFC/beijing.shtml')
# 使用 utf-8 解决文字乱码
rep.encoding = 'utf-8'
# 获取网页内容
content = rep.text

# 使用 xpath 解析
html = etree.HTML(content)
# xpath 解析到城市对应的元素
citys = html.xpath('//div[@class="lqcontentBoxheader"]/ul/li/a')
name_url = {}
for city in citys:
    # 解析元素的文本内容，也就是城市名字
    name = city.xpath('text()')[0]
    # 解析元素的链接，也就是每个城市对应的链接，比如 /textFC/beijing.shtml
    url = city.xpath('@href')[0]
    # 链接完整
    url = 'http://www.weather.com.cn' + url
    name_url[name] = url

record = []

# 依次遍历各个城市对应的网页
for name in name_url:
    url = name_url[name]
    rep = requests.get(url)
    # 使用 utf-8 解决文字乱码
    rep.encoding = 'utf-8'
    # 获取网页内容
    content = rep.text

    # 使用 xpath 解析
    html = etree.HTML(content)

    # 所有日期
    dates = html.xpath('//ul[@class="day_tabs"]/li/text()')
    # 所有日期对应的天气数据

    for index, date in enumerate(dates):
        # date格式为今天周三(3月16日)，我们只要括号里的日期，使用正则提取
        date = re.findall(r'[(](.*?)[)]', date)[0]

        # 当前日期对应的天气数据
        city = html.xpath(
            '//div[@class="hanml"]/div[{}]/div[2]/table/tr[1]/td[2]/a/text()'.
            format(index + 1))[0]
        # 白天天气
        bttq = html.xpath(
            '//div[@class="hanml"]/div[{}]/div[2]/table/tr[1]/td[3]/text()'.
            format(index + 1))[0]
        # 最高气温
        zgqw = html.xpath(
            '//div[@class="hanml"]/div[{}]/div[2]/table/tr[1]/td[5]/text()'.
            format(index + 1))[0]
        # 夜间天气
        yjtq = html.xpath(
            '//div[@class="hanml"]/div[{}]/div[2]/table/tr[1]/td[6]/text()'.
            format(index + 1))[0]
        # 最低气温
        zdqw = html.xpath(
            '//div[@class="hanml"]/div[{}]/div[2]/table/tr[1]/td[8]/text()'.
            format(index + 1))[0]

        record.append([city, date, bttq, yjtq, zgqw, zdqw])

df = pd.DataFrame(record)
df.columns = ['城市', '日期', '白天天气', '夜间天气', '最高气温', '最低气温']
df.to_csv('七日天气预报.csv', index=False)
