from operator import index
import pandas as pd

df = pd.read_csv('ratings.csv').head(300)

comments = df['comment'].values.tolist()

with open('comments.txt', 'w') as f:
    for comment in comments:
        if type(comment) == float:
            continue
        f.write(comment)
