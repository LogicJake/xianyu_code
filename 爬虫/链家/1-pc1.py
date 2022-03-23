from time import sleep

import pandas as pd
import requests
from lxml import etree

qys = {
    '东城区': 'dongcheng',
    '西城区': 'xicheng',
    '朝阳区': 'chaoyang',
    '海淀区': 'haidian',
    '丰台区': 'fengtai',
    '石景山区': 'shijingshan',
    '通州区': 'tongzhou',
    '昌平区': 'changping',
    '大兴区': 'daxing',
    '顺义区': 'shunyi',
    '房山区': 'fangshan',
}

records = {
    '链接': [],
    '所在区': [],
    '小区名称': [],
    '户型': [],
    '面积(平米)': [],
    '朝向': [],
    '装修': [],
    '所在楼层': [],
    '建造年份': [],
    '所在位置': [],
    '总价(万元)': [],
    '单价(元/平米)': [],
}

for qy in qys:
    qy_code = qys[qy]
    base_url = 'https://bj.lianjia.com/ershoufang/' + qy_code

    for page in range(37):
        url = base_url + '/pg{}'.format(page)
        print(url)

        rep = requests.get(url)
        content = rep.text

        root = etree.HTML(content)

        items = root.xpath('//ul[@class="sellListContent"]/li')

        for item in items:
            try:
                url = item.xpath(
                    'div[@class="info clear"]/div[@class="title"]/a/@href')[0]
                name = item.xpath(
                    'div[@class="info clear"]/div[@class="flood"]/div[@class="positionInfo"]/a[1]/text()'
                )[0]
                pos = item.xpath(
                    'div[@class="info clear"]/div[@class="flood"]/div[@class="positionInfo"]/a[2]/text()'
                )[0]
                house_info = item.xpath(
                    'div[@class="info clear"]/div[@class="address"]/div[@class="houseInfo"]/text()'
                )[0]

                total_price = item.xpath(
                    'div[@class="info clear"]/div[@class="priceInfo"]/div[1]/span/text()'
                )[0]

                unit_price = item.xpath(
                    'div[@class="info clear"]/div[@class="priceInfo"]/div[2]/span/text()'
                )[0]

                if len(house_info.split('|')) != 7:
                    continue

                hx, area, cx, zx, lc, year, _ = house_info.split('|')

                if '年' not in year:
                    continue

                hx = hx.strip()
                area = float(area.strip()[:-2])
                cx = cx.strip().replace(' ', '')
                zx = zx.strip()
                lc = lc.strip()
                year = year.strip()[:-1]

                total_price = int(total_price.strip())
                unit_price = int(unit_price.strip()[:-3].replace(',', ''))

            except Exception:
                continue

            records['链接'].append(url)
            records['所在区'].append(qy)
            records['小区名称'].append(name)
            records['户型'].append(hx)
            records['面积(平米)'].append(area)
            records['朝向'].append(cx)
            records['装修'].append(zx)
            records['所在楼层'].append(lc)
            records['建造年份'].append(year)
            records['所在位置'].append(pos)
            records['总价(万元)'].append(total_price)
            records['单价(元/平米)'].append(unit_price)

        print(len(records['链接']))
        sleep(5)

df = pd.DataFrame(records)
df.to_csv('data1.csv', index=False)
