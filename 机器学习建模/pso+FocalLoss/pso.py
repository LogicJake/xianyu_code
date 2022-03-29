import random

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


class PSO:
    def __init__(self, rounds, pop_size, var_num, low_boud, high_bound,
                 X_train, X_test, y_train, y_test):
        self.rounds = rounds  # 迭代的代数
        self.pop_size = pop_size  # 种群大小
        self.var_num = var_num  # 变量个数
        self.low_boud = low_boud  # 变量下限
        self.high_bound = high_bound  # 变量上限

        self.pop_x = np.zeros((self.pop_size, self.var_num))  # 所有粒子的位置
        self.pop_v = np.zeros((self.pop_size, self.var_num))  # 所有粒子的速度
        self.p_best = np.zeros((self.pop_size, self.var_num))  # 每个粒子最优的位置
        self.g_best = np.zeros((1, self.var_num))  # 全局最优的位置

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        # 初始化
        temp = float('-inf')
        for i in range(self.pop_size):
            for j in range(self.var_num):
                self.pop_x[i][j] = random.uniform(self.low_boud[j],
                                                  self.high_bound[j])
                self.pop_v[i][j] = random.uniform(0, 1)
            self.p_best[i] = self.pop_x[i]  # 储存最优的个体
            fit = self.fitness(self.p_best[i])
            if fit > temp:
                self.g_best = self.p_best[i]
                temp = fit

    def fitness(self, vars):
        """
        适应值计算
        """
        # xgb 5个参数
        n_estimators = int(vars[0])
        max_depth = int(vars[1])
        learning_rate = vars[2]
        min_child_weight = vars[3]
        colsample_bytree = vars[4]

        # pso 2个参数
        alpha = vars[5]
        gamma = vars[6]

        def focal_loss_objective(y, p):
            # sigmoid 转为概率
            p = 1.0 / (1.0 + np.exp(-p))

            # focal_loss 一阶导
            grad = p * (1 - p) * (alpha * gamma * y *
                                  (1 - p)**gamma * np.log(p) /
                                  (1 - p) - alpha * y *
                                  (1 - p)**gamma / p - gamma * p**gamma *
                                  (1 - alpha) *
                                  (1 - y) * np.log(1 - p) / p + p**gamma *
                                  (1 - alpha) * (1 - y) / (1 - p))
            # focal_loss 二阶导
            hess = p * (1 - p) * (
                p * (1 - p) *
                (-alpha * gamma**2 * y * (1 - p)**gamma * np.log(p) /
                 (1 - p)**2 + alpha * gamma * y * (1 - p)**gamma * np.log(p) /
                 (1 - p)**2 + 2 * alpha * gamma * y * (1 - p)**gamma /
                 (p * (1 - p)) + alpha * y *
                 (1 - p)**gamma / p**2 - gamma**2 * p**gamma * (1 - alpha) *
                 (1 - y) * np.log(1 - p) / p**2 + 2 * gamma * p**gamma *
                 (1 - alpha) * (1 - y) / (p * (1 - p)) + gamma * p**gamma *
                 (1 - alpha) * (1 - y) * np.log(1 - p) / p**2 + p**gamma *
                 (1 - alpha) * (1 - y) / (1 - p)**2) - p *
                (alpha * gamma * y * (1 - p)**gamma * np.log(p) /
                 (1 - p) - alpha * y * (1 - p)**gamma / p - gamma * p**gamma *
                 (1 - alpha) * (1 - y) * np.log(1 - p) / p + p**gamma *
                 (1 - alpha) * (1 - y) / (1 - p)) + (1 - p) *
                (alpha * gamma * y * (1 - p)**gamma * np.log(p) /
                 (1 - p) - alpha * y * (1 - p)**gamma / p - gamma * p**gamma *
                 (1 - alpha) * (1 - y) * np.log(1 - p) / p + p**gamma *
                 (1 - alpha) * (1 - y) / (1 - p)))

            return grad, hess

        clf = xgb.XGBClassifier(objective=focal_loss_objective,
                                use_label_encoder=False,
                                n_estimators=n_estimators,
                                max_depth=max_depth,
                                learning_rate=learning_rate,
                                min_child_weight=min_child_weight,
                                colsample_bytree=colsample_bytree)

        clf.fit(self.X_train, self.y_train, eval_metric='logloss')
        y_predict = clf.predict_proba(self.X_test)[:, 1]
        fitness = roc_auc_score(self.y_test, y_predict)
        return fitness

    def update_operator(self):
        """
        更新下一时刻的位置和速度
        """
        c1 = 2  # 学习因子，一般为2
        c2 = 2
        w = 0.4  # 自身权重因子
        for i in range(self.pop_size):
            # 更新速度
            self.pop_v[i] = w * self.pop_v[i] + c1 * random.uniform(
                0, 1) * (self.p_best[i] - self.pop_x[i]) + c2 * random.uniform(
                    0, 1) * (self.g_best - self.pop_x[i])
            # 更新位置
            self.pop_x[i] = self.pop_x[i] + self.pop_v[i]
            # 越界保护
            for j in range(self.var_num):
                if self.pop_x[i][j] < self.low_boud[j]:
                    self.pop_x[i][j] = self.low_boud[j]
                if self.pop_x[i][j] > self.high_bound[j]:
                    self.pop_x[i][j] = self.high_bound[j]

            # 更新p_best和g_best
            fitness = self.fitness(self.pop_x[i])
            if fitness > self.fitness(self.p_best[i]):
                self.p_best[i] = self.pop_x[i]
            if fitness > self.fitness(self.g_best):
                self.g_best = self.pop_x[i]

    def search(self):
        fitness_list = []

        for _ in tqdm(range(self.rounds)):
            self.update_operator()
            fitness_list.append(self.fitness(self.g_best))

        return fitness_list
