import pandas as pd

df = pd.read_csv('comments.csv')

comments = df['content'].values.tolist()

with open('豆瓣评论.txt', 'w') as f:
    for comment in comments:
        if type(comment) == float:
            continue
        f.write(comment)
