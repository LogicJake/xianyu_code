"""Scientific Computation Project 4
Your CID here:
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


def WTdist(M, Dh):
    """
    Question 1:
    Compute WT distance matrix, X, given probability matrix, M, and Dh = D^{-1/2}
    """
    a, b = M.shape
    X = np.zeros((a, b))
    A = np.dot(Dh, M.T)
    for i in range(a):
        for j in range(i, b):
            X[i][j] = np.sqrt(np.sum((A[:, i] - A[:, j])**2))
    X = X + X.T
    return X


def WTdist2(M, Dh, Clist):
    """
    Question 2:
    Compute squared distance matrix, Y, given probability matrix, M, Dh = D^{-1/2},
    and community list, Clist
    """

    a = len(Clist)
    Y = np.zeros((a, a))
    for i in range(a):
        for j in range(a):
            Y[i][j] = np.sum(
                (np.sum(M[Clist[i], :], axis=0) -
                 np.sum(M[Clist[j], :], axis=0))**2 * np.diag(Dh)**2)

    return Y


def makeCdict(G, Clist):
    """
    For each distinct pair of communities a,b determine if there is at least one link
    between the communities. If there is, then b in Cdict[a] = a in Cdict[b] = True
    """
    m = len(Clist)
    Cdict = {}
    for a in range(m - 1):
        for b in range(a + 1, m):
            if len(list(nx.edge_boundary(G, Clist[a], Clist[b]))) > 0:
                if a in Cdict:
                    Cdict[a].add(b)
                else:
                    Cdict[a] = {b}

                if b in Cdict:
                    Cdict[b].add(a)
                else:
                    Cdict[b] = {a}
    return Cdict


def analyzeFD():
    """
    Question 4:
    Add input/output as needed

    """

    return None  # modify as needed


def main(t=6, Nc=2):
    """
    WT community detection method
    t: number of random walk steps
    Nc: number of communities at which to terminate the method
    """

    # Read in graph
    A = np.load('data4.npy')
    G = nx.from_numpy_array(A)
    N = G.number_of_nodes()

    # Construct array of node degrees and degree matrices
    k = np.array(list(nx.degree(G)))[:, 1]
    D = np.diag(k)
    Dinv = np.diag(1 / k)
    Dh = np.diag(np.sqrt(1 / k))

    P = Dinv.dot(A)  # transition matrix
    M = np.linalg.matrix_power(P, t)  # P^t
    X = WTdist(M, Dh)  # Q1: function to be completed

    # Initialize community list
    Clist = []
    for i in range(N):
        Clist.append([i])

    Y = WTdist2(M, Dh, Clist)  # Q2: function to be completed

    m = len(Clist)  # number of communities
    m_total = len(Clist)  # 这个值不变，代表原始的聚类数量

    Cdict = makeCdict(G, Clist)

    # make list of community sizes
    L = []
    for l in Clist:
        L.append(len(l))

    # Q3: Add code here, use/modify/remove code above as needed
    # 核心思想是不删除元素，最后检查删除空聚类即可
    S = 10000 * np.ones((m_total, m_total))
    for a in range(m_total):
        for b in range(a + 1, m_total):
            if a in Cdict:
                if b in Cdict[a]:
                    S[a][b] = 1 / N * L[a] * L[b] / (L[a] + L[b]) * Y[a][b]

    while m > Nc:
        smin = float("inf")
        for a in range(m_total - 1):
            la = L[a]

            # 聚类空了跳过，保证被合并的聚类不会影响后续合并过程
            if la == 0:
                continue

            for b in range(a + 1, m_total):
                lb = L[b]
                # 聚类空了跳过，保证被合并的聚类不会影响后续合并过程
                if lb == 0:
                    continue

                if a in Cdict:
                    if b in Cdict[a]:
                        s = S[a][b]
                        if s < smin:
                            amin = a
                            bmin = b
                            smin = s

        print(m, amin, bmin, smin)
        # 合并聚类
        # 更新S
        # 新建一个S：provided that s ab , s aj , and s bj had been computed prior to merging.
        S_new = S.copy()
        for j in Cdict[amin]:
            lj = L[j]
            lamin = L[amin]
            lbmin = L[bmin]

            S_new[amin][j] = ((lamin + lj) * S[amin][j] +
                              (lbmin + lj) * S[bmin][j] -
                              lj * S[amin][bmin]) / (lamin + lbmin + lj)
        # 修改完毕，新的S
        S = S_new.copy()

        # bmin聚类合并到amin
        Clist[amin] = Clist[amin] + Clist[bmin]
        # bmin聚类变为空
        Clist[bmin] = []
        # 聚类个数-1
        m -= 1
        # 重新计算连通性
        Cdict = makeCdict(G, Clist)
        # 重新计算聚类size
        L = []
        for l in Clist:
            L.append(len(l))

        # print(m, amin, bmin, smin)
        # print(Clist)
        # print(L)
        # print(Cdict)
        # exit()

    Lfinal = [l for l in Clist if len(l) != 0]
    return Lfinal


if __name__ == '__main__':
    t = 6
    out = main(t)
    print(out)
