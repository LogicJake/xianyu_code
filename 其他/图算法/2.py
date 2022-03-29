import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from node2vec import Node2Vec
from sklearn.cluster import KMeans

# 读取本地邻接表
graph = nx.read_adjlist('email.txt', create_using=nx.DiGraph)

# 初始化node2vec
node2vec = Node2Vec(
    graph,
    dimensions=2,  # 最后得到的向量维度，不能更改，因为可视化图是2维的
    walk_length=30,  # 可以改
    num_walks=100,  # 可以改
    workers=4,  # 多进程数量
)

# 学习网络中节点的向量表示，这些参数都可以改，可以多改改比较效果，多花点图
model = node2vec.fit(window=10, min_count=1, batch_words=4)

vecs = []
# 遍历得到每个节点的向量表示
for word in model.wv.index_to_key:
    vec = list(model.wv[word])
    vecs.append(vec)
vecs = np.array(vecs)

# 进行聚类，聚类数量为3，可以修改n_clusters
estimator = KMeans(n_clusters=3)
# 开始聚类
estimator.fit(vecs)
# 得到每个点的聚类标签
labels = estimator.labels_

# 绘制散点图
plt.scatter(vecs[:, 0], vecs[:, 1], c=labels)
plt.show()
