import pandas as pd

df = pd.read_csv('bitcoin.txt')
df.to_csv('bitcoin2.txt', sep=' ', index=False)
