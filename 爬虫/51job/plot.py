import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['figure.figsize'] = (10.0, 6.0)  # 设置figure_size尺寸

os.makedirs('fig', exist_ok=True)

df = pd.read_excel('data.xls')


def func(x):
    x = x[:-3]

    start, end = x.split('-')

    start = float(start)
    end = float(end)

    return (start + end) / 2


df['工资待遇'] = df['工资待遇'].apply(func)

for f in ['工作地点', '经验要求', '学历要求', '需求人数']:
    tmp = df.groupby(f)['工资待遇'].mean().round(3).reset_index()
    tmp = tmp.sort_values(['工资待遇'], ascending=False)

    if f == '工作地点':
        tmp = tmp.head(15)

    plt.cla()
    plt.clf()
    plt.bar(tmp[f], tmp['工资待遇'])
    plt.xlabel(f)
    plt.ylabel('工资待遇(万/月)')
    plt.title('不同{}工资待遇差异'.format(f))

    for a, b in zip(tmp[f], tmp['工资待遇']):
        plt.text(a, b, b, ha='center', va='bottom')

    plt.savefig(os.path.join('fig', '{}_bar.png'.format(f)))
