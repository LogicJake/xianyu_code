import datetime
import time

import requests

# 间隔多少秒
sleep_sec = 3
# 是否测试，建议每次运行前先测试一遍
test = True

token = 'SCT130333T9bkO9Ki5FzD7Qj3rjvYrEtmk'

today = datetime.date.today()
all_days = 14
url = 'https://i.hzmbus.com/webh5api/manage/query.book.info.data'

while True:
    try:

        buys = []

        for i in range(all_days):
            date = (today + datetime.timedelta(days=i)).strftime('%Y-%m-%d')

            payload = {
                "bookDate": date,
                "lineCode": "ZHOHKG" if test else "HKGZHO",
                "appId": "HZMBWEB_HK",
                "joinType": "WEB",
                "version": "2.7.202203.1092",
                "equipment": "PC"
            }

            rep = requests.post(url, json=payload)
            result = rep.json()

            # 没到时间，不可预约
            if 'responseData' not in result:
                raise Exception(str(result))

            if len(result['responseData']) == 0:
                break

            for banci in result['responseData']:
                beginTime = banci['beginTime']
                num = banci['maxPeople']

                if num == 0:
                    continue
                else:
                    buys.append([date, beginTime, num])

        if len(buys) == 0:
            continue

        msg = '总共有{}个班次可以购票\n\n'.format(len(buys))
        for buy in buys:
            date, beginTime, num = buy
            msg += '日期：{}，发车时间：{}，剩余人数：{}\n\n'.format(date, beginTime, num)

        data = {
            'title': '港珠澳大桥穿梭巴士（珠海-香港）' if test else '港珠澳大桥穿梭巴士（香港-珠海）',
            'desp': msg
        }

        print(msg.replace('\n\n', '\n'))

        url = 'https://sctapi.ftqq.com/{}.send'.format(token)
        rep = requests.post(url, data=data)

        if test:
            break

        time.sleep(sleep_sec)

    except Exception as e:
        print(e)
