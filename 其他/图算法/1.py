import collections

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os

os.makedirs('fig', exist_ok=True)


# 读取本地邻接表
def read_adjacency_table(path):
    # 所有的节点
    nodes = set()
    # 所有的边
    edges = []

    # 打开本地文件
    with open(path, 'r') as f:
        # 读取所有行
        lines = f.readlines()
        for line in lines:
            line = line.strip()

            # 根据空格分割
            node1, node2 = line.split(' ')

            nodes.add(node1)
            nodes.add(node2)
            edges.append([node1, node2])

    nodes = list(nodes)
    # 根据节点id排序
    nodes.sort()

    return nodes, edges


# 转成邻接矩阵
def to_adjacency_matrix(nodes, edges):
    # 节点数量
    nlen = len(nodes)
    index = dict(zip(nodes, range(nlen)))

    # nlen * nlen 大小的空矩阵
    A = np.full((nlen, nlen), 0)
    # 遍历所有边
    for u, v in edges:
        # 两个之间有边，则值为1
        A[index[u], index[v]] = 1

    return A


def build_network(nodes, edges):
    # 初始化空的有向图
    g = nx.DiGraph()

    # 添加节点
    g.add_nodes_from(nodes)
    # 添加边
    g.add_edges_from(edges)

    return g


# 可视化图
def plot_network(g, save_path='plot.png'):
    plt.clf()
    # 初始化布局
    pos = nx.spring_layout(g)
    # 绘制基本的节点和边
    nx.draw(g, node_size=300, pos=pos)
    # 绘制节点id
    nx.draw_networkx_labels(g, pos=pos)
    # 保存到本地
    plt.savefig(os.path.join('fig', save_path))


def plot_bar(degree_cnt, title, xlabel, ylabel):
    plt.clf()

    # 所有的度和对应的数量
    deg, cnt = zip(*degree_cnt.items())

    # 画柱状图
    plt.bar(deg, cnt)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(os.path.join('fig', title + '.png'))


def plot_degree(nodes, edges):
    in_degree = {}
    out_degree = {}
    degree = {}

    # 初始化为0
    for node in nodes:
        in_degree[node] = 0
        out_degree[node] = 0
        degree[node] = 0

    # 计算出度，入度和度
    for u, v in edges:
        out_degree[u] += 1
        in_degree[v] += 1
        degree[u] += 1
        degree[v] += 1

    # 统计度值出现的次数
    degree_cnt = collections.Counter([in_degree[node] for node in nodes])
    # 绘制入度分布图
    plot_bar(degree_cnt,
             title="In Degree Histogram",
             xlabel="In Degree",
             ylabel="Count")
    # 绘制出度分布图
    degree_cnt = collections.Counter([out_degree[node] for node in nodes])
    plot_bar(degree_cnt,
             title="Out Degree Histogram",
             xlabel="Out Degree",
             ylabel="Count")
    # 绘制度分布图
    degree_cnt = collections.Counter([degree[node] for node in nodes])
    plot_bar(degree_cnt,
             title="Degree Histogram",
             xlabel="Degree",
             ylabel="Count")


# 广度遍历 寻找两个节点之间的最短路径
def shortest_path(g, source, target):
    # 判断节点存不存在
    if source not in g.nodes():
        print('节点{}不存在'.format(source))
        return None
    if target not in g.nodes():
        print('节点{}不存在'.format(target))
        return None

    # path = nx.dijkstra_path(g, source, target)

    adj = g._adj  # 节点的相邻节点集合

    # 首尾节点一样
    if target == source:
        return [source]

    # 待搜寻的队列
    search_queue = []
    # 加入source节点，和当前到source的路径
    search_queue.append((source, [source]))
    searched = []

    path = []

    while search_queue:
        # 先从队列中取出一个节点
        node, node_path = search_queue.pop(0)
        # 该节点如果没被查找过，则进行检查
        if node not in searched:
            # 如果该节点，为target
            if node == target:
                # 打印最短路径
                path = node_path
                break
            # 如果当前节点不是target，则把其邻居入队并把该节点加入被检查列表中
            else:
                for next_node in adj[node]:
                    search_queue.append((next_node, node_path + [next_node]))

                searched.append(node)

    return path


def dfs(x, target, all_path, path, adj):
    if x == target:
        all_path.append(path[:])
        return

    for y in adj[x]:
        path.append(y)
        dfs(y, target, all_path, path, adj)
        path.pop()


# 寻找所有的路径
def all_path(g, source, target):
    all_path = []
    path = []
    adj = g._adj  # 节点的相邻节点集合

    path.append(source)
    dfs(source, target, all_path, path, adj)

    return all_path


# 判断图是否有回路
def has_circle(nodes, edges):
    indeg = {}
    adj = {}

    for node in nodes:
        indeg[node] = 0
        adj[node] = set()

    for info in edges:
        adj[info[0]].add(info[1])
        indeg[info[1]] += 1

    q = [node for node in nodes if indeg[node] == 0]
    visited = 0

    while q:
        visited += 1
        u = q.pop(0)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)

    res = visited < len(nodes)
    return res


def components(g):
    adj = {}

    for node in g.nodes:
        adj[node] = set()

    for info in g.edges:
        adj[info[0]].add(info[1])
        adj[info[1]].add(info[0])

    component = []
    seen = set()
    # 遍历所有节点
    for u in g.nodes:
        # 已经被访问过就跳过
        if u in seen:
            continue
        current = walk(adj, u, seen)
        component.append(current)

    return component


def walk(adj, start, seen):
    nodes = [start]
    current = []

    while nodes:
        u = nodes.pop(0)
        current.append(u)
        seen.add(u)

        for v in adj[u]:
            if v in seen or v in nodes:
                continue

            nodes.append(v)

    return current


def print_meau():
    meau = '\n*******************************\n'
    meau += '1. 输出图的邻接矩阵\n'
    meau += '2. 可视化图结构\n'
    meau += '3. 画出节点度的分布图\n'
    meau += '4. 找最短路径\n'
    meau += '5. 找所有路径\n'
    meau += '6. 判断图是否有回路\n'
    meau += '7. 判断图是否存在多个连通分量\n'
    meau += '8. 退出程序\n'
    meau += '*******************************\n'
    print(meau)


if __name__ == '__main__':
    path = None
    while True:
        try:
            path = input('请输入邻接表txt路径：')

            if not os.path.exists(path):
                print('无效文件路径')
            else:
                break

        except Exception:
            print('无效输入')

    nodes, edges = read_adjacency_table(path)
    g = build_network(nodes, edges)

    while True:

        while True:
            try:
                print_meau()
                choice = int(input('请输入要执行的功能编号：'))

                if choice < 1 or choice > 8:
                    print('无效输入')
                else:
                    break
            except Exception:
                print('无效输入')

        if choice == 1:
            A = to_adjacency_matrix(nodes, edges)
            print(A)
        elif choice == 2:
            path = 'plot.png'
            plot_network(g, save_path=path)
            print('已保存到本地，路径为：{}'.format(
                os.path.abspath(os.path.join('fig', path))))
        elif choice == 3:
            plot_degree(nodes, edges)
            print('已保存到本地文件夹下，路径为：{}'.format(os.path.abspath('fig')))

        elif choice == 4:
            source = input('请输入起始节点：')
            target = input('请输入终止节点：')

            path = shortest_path(g, source, target)
            print('节点{}和节点{}之间的最短路径为：{}'.format(source, target, path))
        elif choice == 5:
            if has_circle(nodes, edges):
                print('当前图中存在回路，无法寻找所有的路径')
            else:
                source = input('请输入起始节点：')
                target = input('请输入终止节点：')

                path = all_path(g, source, target)
                print('节点{}和节点{}之间的所有路径为：{}'.format(source, target, path))
        elif choice == 6:
            if has_circle(nodes, edges):
                print('当前图中存在回路')
            else:
                print('当前图中不存在回路')
        elif choice == 7:
            component = components(g)
            print('当前图有{}个联通分量，分别是：{}'.format(len(component), component))
        elif choice == 8:
            break
