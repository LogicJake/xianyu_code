from operator import index
import random
import pandas as pd

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["SimHei"]


def simulation(nums, stop):
    # stop之前的最大值
    max_before = max(nums[:stop])

    choise = 0
    # 从stop开始做决定
    for num in nums[stop:]:
        # 找到一个比之前观测的好的，甚至一样好的就停止
        if num >= max_before:
            choise = num
            break
    # 全局最优
    max_total = max(nums)

    # 如果选到的恰好是全局最优，返回1，成功
    if max_total == choise:
        return 1
    # 如果没有选到，返回0，失败
    else:
        return 0


lifes = []

# 产生1000次模拟人生
for _ in range(1000):
    # 产生20个候选人
    nums = []

    for _ in range(20):
        # 数值代表相亲的候选人评分，越高越好
        nums.append(random.randint(1, 100))

    lifes.append(nums)

# 开始模拟
cnt = [0 for _ in range(19)]

# 对每个停止时间进行模拟
for stop in range(1, 20):
    for life in lifes:
        # 记录成功次数
        cnt[stop - 1] += simulation(life, stop)

df = pd.DataFrame({'停止时间': range(1, 20), '找到真命天子的次数': cnt})
df.to_csv('1.csv', index=False)
# 画图
plt.plot(range(1, 20), cnt)
plt.xlabel('停止时间')
plt.ylabel('找到真命天子的次数')
plt.savefig('1.png')
plt.show()
