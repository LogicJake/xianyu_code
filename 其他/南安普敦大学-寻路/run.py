#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:13:29 2022

@author: tvp1e19
"""

import numpy as np
import pandas as pd

LNUM = 99999999999999999999999


def time_spend(ship1, ship2):
    a = ship2[3]**2 + ship2[4]**2 - (ship1[3]**2 + ship1[4]**2)
    b = 2 * ((ship2[1] - ship1[1]) * ship2[3] +
             (ship2[2] - ship1[2]) * ship2[4])
    c = (ship2[1] - ship1[1])**2 + (ship2[2] - ship1[2])**2

    # 求根公式，根据delta的值分为三种情况
    delta = b**2 - 4 * a * c
    if delta < 0:
        return LNUM
    elif delta > 0:
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        t2 = (-b - np.sqrt(delta)) / (2 * a)

        ans = []
        # 只要解是正的
        if t1 > 0:
            ans.append(t1)
        if t2 > 0:
            ans.append(t2)
        # 没有正数解，无解
        if len(ans) == 0:
            return LNUM
        # 多个正数解，返回最小的
        else:
            return min(ans)

    else:
        # 一个解的情况，检查是不是正数
        if -b / (2 * a) < 0:
            return LNUM
        else:
            return -b / (2 * a)


def read_srp_input_data(csv_file):
    '''
    read the given data file by pandas
    '''
    input_data = pd.read_csv(csv_file)

    return input_data


#######################
# main function
def findPath(input_data):
    '''
    Keyword arguments: the input_data
        
    Returns: unvisited_ships--list of ships which cannot be visited
            visited_ships -- list of visited cruise ships
            total_time--total time to visit all ships
        
    '''
    total_time = 0
    visited_ships = []
    unvisited_ships = []

    # 二维数组，每一行代表一条船，第0行是搜救船。
    # 第0列是船名称（无用），
    # 第1列 x-coordinate
    # 第2列 y-coordinate
    # 第3列 x-speed
    # 第4列 y-speed
    all_ships = input_data.values

    # 初始情况，所有船只都是unvisited
    unvisited_ships = [i for i in range(1, input_data.shape[0])]
    while True:
        mt = LNUM
        next = None

        # 依次计算到达未访问船只的时间
        for ship in unvisited_ships:
            t = time_spend(all_ships[0], all_ships[ship])
            # 时间更小则更新
            if t < mt:
                mt = t
                next = ship

        # 如果检查结束最小时间还是LNUM，说明没有找到可以到达的船只，跳出循环
        if mt == LNUM:
            break

        # 根据pdf公式计算船只mt时刻后的位置
        for ship in unvisited_ships:
            all_ships[ship][1] = all_ships[ship][1] + mt * all_ships[ship][3]
            all_ships[ship][2] = all_ships[ship][2] + mt * all_ships[ship][4]

        # 将救援船的位置更新为刚访问到的船只的位置，因为两者碰面了
        all_ships[0][1] = all_ships[next][1]
        all_ships[0][2] = all_ships[next][2]

        # 更新 visited_ships
        visited_ships.append(next)
        # 更新 unvisited_ships
        unvisited_ships.remove(next)
        # 累计时间
        total_time += mt

    return unvisited_ships, visited_ships, total_time


# You can write additional functions in needed.

input_data = read_srp_input_data('test_data_1.csv')
unvisited_ships, visited_ships, total_time = findPath(input_data)
print(unvisited_ships, visited_ships, total_time)
