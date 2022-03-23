import pandas as pd
from lxml import etree
from selenium import webdriver
from tqdm import tqdm
import time

money_list = []
place_list = []
jy_list = []
edu_list = []
num_list = []

try:
    driver = webdriver.PhantomJS()
except Exception:
    driver = webdriver.PhantomJS(executable_path='phantomjs.exe')

for page in tqdm(range(20)):
    driver.get(
        'https://search.51job.com/list/000000,000000,0000,00,9,99,%25E9%2594%2580%25E5%2594%25AE,2,{}.html'
        .format(page))

    source = driver.page_source

    # source = ''
    # with open('1.html', 'r') as f:
    #     source = f.read()

    html = etree.HTML(source)

    moneys = []
    infos = []

    for content in html.xpath('//p[@class="info"]'):
        try:
            money = content.xpath('span[1]/text()')

            if len(money) == 0:
                continue

            info = content.xpath('span[2]/text()')

            money = money[0]
            info = info[0]

            m, danwei = money.split('/')
            start, end = m[:-1].split('-')

            start = float(start)
            end = float(end)

            if m[-1] == '千':
                start = start / 10
                end = end / 10

            if danwei == '年':
                start = start / 12
                end = end / 12

            start = round(start, 1)
            end = round(end, 1)

            if len(info.split('|')) != 4:
                continue

            place, jy, edu, num = info.split('|')

            place = place.strip().split('-')[0]
            jy = jy.strip()
            edu = edu.strip()
            num = num.strip()[1:]

            place_list.append(place)
            jy_list.append(jy)
            edu_list.append(edu)
            num_list.append(num)
            money_list.append('{}-{}万/月'.format(start, end))

        except Exception:
            continue

    time.sleep(2)

driver.close()
driver.quit()

df = pd.DataFrame({
    '工资待遇': money_list,
    '工作地点': place_list,
    '经验要求': jy_list,
    '学历要求': edu_list,
    '需求人数': num_list
})

df.to_excel('data.xls', index=False)
