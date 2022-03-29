#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:13:29 2022

@author: tvp1e19
"""

from copy import copy
import pandas as pd
import numpy as np


def read_srp_input_data(csv_file):
    '''
    read the given data file by pandas
    '''
    input_data = pd.read_csv(csv_file)

    return input_data


# 计算两个船的追击时间
def calculate_time(from_x, from_y, from_vx, from_vy, to_x, to_y, to_vx, to_vy):
    # 搜救船的速度
    S = np.sqrt(from_vx**2 + from_vy**2)
    # 二元一次函数的a值
    a = to_vx**2 + to_vy**2 - S**2
    # 二元一次函数的b值
    b = 2 * ((to_x - from_x) * to_vx + (to_y - from_y) * to_vy)
    # 二元一次函数的c值
    c = (to_x - from_x)**2 + (to_y - from_y)**2

    # 根据二元一次方程求根公式判断有没有解
    delta = b**2 - 4 * a * c
    # delta 小于0 无解
    if delta < 0:
        return float('inf')
    # delta 等于0 1个解
    if delta == 0:
        # 计算时间
        t = -b / (2 * a)
        # 时间小于0，无意义
        if t < 0:
            return float('inf')
        # 时间正常
        else:
            # 返回时间
            return t
    # delta 大于0 2个解
    if delta > 0:
        # 计算追击时间t1
        t1 = (-b + np.sqrt(delta)) / (2 * a)
        # 计算追击时间t2
        t2 = (-b - np.sqrt(delta)) / (2 * a)

        # 如果两个解都是负的，无解
        if t1 < 0 and t2 < 0:
            return float('inf')
        # 如果两个解都是正的，返回最小的
        elif t1 > 0 and t2 > 0:
            return min(t1, t2)
        # t1是正的就返回t1
        elif t1 > 0:
            return t1
        # t2是正的就返回t2
        else:
            return t2


def find_next_ship(data, unvisited_ships):
    # 最小追击时间，初始化为无穷大
    min_t = float('inf')
    # 确定追击的船的编号，初始化为-1
    next_ship = -1

    # 搜救船的信息
    # 坐标x
    from_x = data[0]['x-coordinate']
    # 坐标y
    from_y = data[0]['y-coordinate']
    # 速度x
    from_vx = data[0]['x-speed']
    # 速度y
    from_vy = data[0]['y-speed']

    # 遍历还未访问的船
    for ship in unvisited_ships:
        # 当前船的相关信息
        # 坐标x
        to_x = data[ship]['x-coordinate']
        # 坐标y
        to_y = data[ship]['y-coordinate']
        # 速度x
        to_vx = data[ship]['x-speed']
        # 速度y
        to_vy = data[ship]['y-speed']

        # 计算救援船到当前船的追击时间
        t = calculate_time(from_x, from_y, from_vx, from_vy, to_x, to_y, to_vx,
                           to_vy)
        # 如果这条船追击时间更短
        # unvisited_ships的编号是从小到大的，而且只有小于更新，这样保证了出现同样时间的时候，返回编号更小的船只
        if t < min_t:
            # 更新最短时间
            min_t = t
            # 更新确定追击的船只
            next_ship = ship

    # 返回遍历完成之后需要追击的船只和追击时间
    return min_t, next_ship


def update_status(data, t, next_ship, unvisited_ships):
    # 拷贝原有数据
    data_ = copy(data)

    # 其他游船恒定方向和速度前进，计算位置
    for ship in unvisited_ships:
        # x = x + t*vx
        data_[ship]['x-coordinate'] = data[ship][
            'x-coordinate'] + t * data[ship]['x-speed']
        # y = y + t*vy
        data_[ship]['y-coordinate'] = data[ship][
            'y-coordinate'] + t * data[ship]['y-speed']

    # 支援船直接和next_ship会和，更新为next_ship的x和y
    data_[0]['x-coordinate'] = data_[next_ship]['x-coordinate']
    data_[0]['y-coordinate'] = data_[next_ship]['y-coordinate']

    return data_


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

    data = {}
    # 转换成字典，方便查询
    for i in range(input_data.shape[0]):
        # 初始化为空字典
        data[i] = {}

        # 序号0是救援船跳过，不加入unvisited_ships
        if i != 0:
            unvisited_ships.append(i)

        # 将船只的相关信息放到对应的字典里面去
        for f in ['x-coordinate', 'y-coordinate', 'x-speed', 'y-speed']:
            data[i][f] = input_data.loc[i, f]

    # 循环遍历
    while True:
        # 根据贪心策略找到需要到达的下一条船
        min_t, next_ship = find_next_ship(data, unvisited_ships)

        # mint是无穷大，说明当前未访问的船只都追不上或者没有待追击的船只，退出循环
        if min_t == float('inf'):
            break

        # 更新 min_t 之后各船的位置
        data = update_status(data, min_t, next_ship, unvisited_ships)

        # 累积总时间
        total_time += min_t
        # next_ship 标记为已访问
        visited_ships.append(next_ship)
        # next_ship 从未访问删除
        unvisited_ships.remove(next_ship)

    return unvisited_ships, visited_ships, total_time


# You can write additional functions in needed.

if __name__ == "__main__":
    input_data = read_srp_input_data('./test_data_1.csv')
    unvisited_ships, visited_ships, total_time = findPath(input_data)
    print(unvisited_ships, visited_ships, total_time)
