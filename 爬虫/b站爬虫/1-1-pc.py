import requests
import pandas as pd

records = {
    'mid': [],
    'duration': [],
    'id': [],
    'author': [],
    'description': [],
    'title': []
}

for page in range(1, 5):
    url = f'https://s.search.bilibili.com/cate/search?search_type=video&view_type=hot_rank&order=scores&cate_id=201&page={page}&pagesize=100&time_from=20220324&time_to=20220331'

    rep = requests.get(url)
    rep = rep.json()

    for video in rep['result']:
        for k in records:
            records[k].append(video[k])

df = pd.DataFrame(records)
df.to_csv('data1.csv', encoding='utf_8_sig', index=False)
