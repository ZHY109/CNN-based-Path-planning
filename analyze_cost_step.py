import numpy as np
import heapq  # 用于优先队列
import cv2
import math
import random
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import pandas as pd

# 定义A*算法相关类和函数
class Node:
    def __init__(self, position, g_cost=0, h_cost=0, parent=None):
        self.position = position  # (x, y)坐标
        self.g_cost = g_cost      # 从起点到当前节点的实际代价
        self.h_cost = h_cost      # 启发式估计到终点的代价
        self.f_cost = g_cost + h_cost  # 总代价
        self.parent = parent      # 父节点

    def __lt__(self, other):
        # 用于优先队列比较，f_cost较小的节点优先
        return self.f_cost < other.f_cost

def heuristic(a, b):
    # 曼哈顿距离作为启发式函数
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def a_star(start, goal, obstacles):
    # 将坐标转换为0索引
    start_pos = (start[0]-1, start[1]-1)
    goal_pos = (goal[0]-1, goal[1]-1)
    map_size = 25
    # 初始化开放列表和关闭列表
    open_list = []
    closed_list = set()
    
    # 创建起始节点并加入开放列表
    start_node = Node(start_pos)
    heapq.heappush(open_list, start_node)
    
    # 记录已访问节点的字典
    visited = {start_pos: start_node}
    
    while open_list:
        # 取出f_cost最小的节点
        current_node = heapq.heappop(open_list)
        current_pos = current_node.position
        
        # 检查是否到达目标
        if current_pos == goal_pos:
            # 回溯路径
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]  # 反转得到从起点到终点的路径
        
        # 将当前节点加入关闭列表
        closed_list.add(current_pos)
        
        # 生成邻居节点（上下左右四个方向）
        neighbors = [
            (current_pos[0]+1, current_pos[1]),  # 右
            (current_pos[0]-1, current_pos[1]),  # 左
            (current_pos[0], current_pos[1]+1),  # 下
            (current_pos[0], current_pos[1]-1),   # 上
            (current_pos[0]-1, current_pos[1]-1),
            (current_pos[0]-1, current_pos[1]+1),
            (current_pos[0]+1, current_pos[1]+1),
            (current_pos[0]+1, current_pos[1]-1)
        ]
        
        for neighbor_pos in neighbors:
            # 检查邻居是否在地图范围内
            if (0 <= neighbor_pos[0] < map_size and 
                0 <= neighbor_pos[1] < map_size):
                # 检查邻居是否是障碍物或已经在关闭列表中
                if (neighbor_pos not in closed_list and 
                    neighbor_pos not in obstacles):
                    # 计算g_cost和h_cost
                    g_cost = current_node.g_cost + 1  # 假设每步代价为1
                    h_cost = heuristic(neighbor_pos, goal_pos)
                    f_cost = g_cost + h_cost
                    
                    # 如果邻居已经在开放列表中，检查是否需要更新
                    if neighbor_pos in visited:
                        existing_node = visited[neighbor_pos]
                        if f_cost < existing_node.f_cost:
                            existing_node.g_cost = g_cost
                            existing_node.f_cost = f_cost
                            existing_node.parent = current_node
                    else:
                        # 创建新节点并加入开放列表
                        neighbor_node = Node(
                            neighbor_pos, 
                            g_cost, 
                            h_cost, 
                            current_node
                        )
                        heapq.heappush(open_list, neighbor_node)
                        visited[neighbor_pos] = neighbor_node
    
    # 如果开放列表为空且未找到路径，返回空列表
    return []
# 计算路径的总step数和总距离代价
def calculate_cost(path):
    total_step = 0
    total_distance_cost = 0
    diagonal_steps = 0
    for i in range(len(path) - 1):
        current_pos = path[i]
        next_pos = path[i + 1]

        if abs(next_pos[0] - current_pos[0]) + abs(next_pos[1] - current_pos[1]) == 1:
            total_step += 1
            total_distance_cost += 1
        else:
            total_step += 2
            total_distance_cost += math.sqrt(2)
            diagonal_steps+=1

    return total_step, total_distance_cost,diagonal_steps

# 主程序中添加路径规划和绘制
def main(train):
    if train:
        pass
    else:
        for case_num in range(1,4):
            start = (25, 1)
            goal = (1, 25)


            # 初始化参数（与原始代码一致）
            map_size = 25
            grid = np.zeros((map_size, map_size))
            grid_answer = np.zeros((map_size, map_size))
            grid_obs = np.zeros((map_size, map_size))
            CELL_SIZE = 30
            WINDOW_SIZE = (map_size*CELL_SIZE, map_size*CELL_SIZE)
            
            # 创建空白图像
            image = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            image_answer = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            image_obs = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            image_all = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255

            # 颜色定义 (BGR格式)
            COLORS = {
                0: (255, 255, 255),    # 空地
                1: (0, 0, 0),          # 障碍物
                2: (0, 255, 0),        # 起点 (绿色)
                3: (0, 255, 255),      # 终点 (黄色)
                4: (0, 0, 255)         # 路径 (蓝色)
            }
            
            # 数据预处理
            def prepare_data(case_num):
                # 填充障碍物
                obstacles = set()
                path = pd.read_csv(f"./inference/mask_sample_{case_num-1}.csv",header=None)
                # print(path.shape)
                csv_obs = 1- path.values
                obs = []
                for y in range(len(csv_obs)):
                    for x in range(len(csv_obs[0])):
                        if csv_obs[y][x]==1:
                            obs.append((y,x))
                for (x, y) in obs:
                    grid_x, grid_y = x, y
                    obstacles.add((grid_x, grid_y))
                    grid_obs[x][y]=1
                
                if case_num == 1:
                    obs = [
                        [1, 2], [1, 3], [2, 4], [2, 5],
                        [3, 6], [4, 8], [11, 20], [13, 12],
                        [13, 13], [13, 15]
                    ]
                elif case_num == 2:
                    obs = [
                        [1, 2], [2, 3], [2, 4], [2, 5], [2, 6],
                        [3, 6], [4, 6], [4, 8], [5, 6],
                        [6, 5], [6, 6], [6, 7], [6, 8], [6, 9], [6, 10],
                        [7, 10], [8, 11], [9, 12],
                        [10, 13], [10, 12], [11, 15], [11, 20],
                        [18, 14], [18, 15], [18, 20], [18, 21], [18, 25],
                        [22, 2], [22, 3], [22, 5]
                    ]
                elif case_num == 3:
                    obs = [
                        [1, 2], [2, 3], [2, 4], [2, 5], [2, 6],
                        [3, 6], [4, 6], [4, 8], [5, 6],
                        [6, 10], [6, 9], [6, 8], [6, 7], [6, 6], [6, 5],
                        [7, 10], [8, 11], [9, 12],
                        [10, 12], [10, 13], [11, 15], [11, 20],
                        [12, 13], [12, 14], [12, 15],
                        [13, 6], [13, 7], [13, 8], [13, 10], [13, 11], [13, 12],
                        [13, 13], [13, 17], [13, 18], [13, 20], [13, 21], [13, 24],
                        [14, 6], [14, 8], [14, 10], [14, 11], [14, 12], [14, 13],
                        [14, 17], [14, 18], [14, 20], [14, 21], [14, 24],
                        [16, 16], [16, 17], [16, 18], [16, 20], [16, 22], [16, 23],
                        [18, 5], [18, 6], [18, 7], [18, 16], [18, 20], [18, 21], [18, 22],
                        [19, 13], [19, 17], [19, 18], [19, 20], [19, 21], [19, 24],
                        [22, 5], [22, 7], [22, 8], [22, 22], [22, 24], [22, 25]
                    ]
                else:
                    obs = []

                for (x, y) in obs:
                    grid_x, grid_y = x-1, y-1
                    grid[grid_x][grid_y] = 1  
               

                # 设置起点终点
                start_x, start_y = start[0]-1, start[1]-1
                grid[start_y][start_x] = 2
                goal_x, goal_y = goal[0]-1, goal[1]-1
                grid[goal_y][goal_x] = 3

                grid_obs[start_y][start_x] = 2
                grid_obs[goal_y][goal_x] = 4
                return obstacles
            
            obstacles = prepare_data(case_num)
            
            # 运行A*算法寻找路径
            path = a_star(start, goal, obstacles)
            
            if path:
                for (x, y) in path:
                    grid_answer[x,y]=3
                steps, cost,diagonal_steps = calculate_cost(path)
                print(f"steps:{steps} (diagonal_steps:{diagonal_steps}), cost:{cost}")

            if path:
                for (y, x) in path:
                    rect_start = (x*CELL_SIZE, y*CELL_SIZE)
                    rect_end = ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE)
                    cv2.rectangle(image, rect_start, rect_end, COLORS[3], -1)
                    cv2.rectangle(image_all, rect_start, rect_end, COLORS[3], -1)

            # 绘制网格
            for y in range(map_size):
                for x in range(map_size):
                    cell_value = grid[y][x]
                    if cell_value > 0:
                        # 绘制单元格
                        rect_start = (x*CELL_SIZE, y*CELL_SIZE)
                        rect_end = ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE)
                        cv2.rectangle(image, rect_start, rect_end, COLORS[cell_value], -1)

            for y in range(map_size):
                for x in range(map_size):
                    cell_value = grid_answer[y][x]
                    if cell_value > 0:
                        # 绘制单元格
                        rect_start = (x*CELL_SIZE, y*CELL_SIZE)
                        rect_end = ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE)
                        cv2.rectangle(image_answer, rect_start, rect_end, COLORS[cell_value], -1)

            for y in range(map_size):
                for x in range(map_size):
                    cell_value = grid_obs[y][x]
                    if cell_value > 0:
                        # 绘制单元格
                        rect_start = (x*CELL_SIZE, y*CELL_SIZE)
                        rect_end = ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE)
                        cv2.rectangle(image_obs, rect_start, rect_end, COLORS[cell_value], -1)
                        cv2.rectangle(image_all, rect_start, rect_end, COLORS[cell_value], -1)

            # 绘制网格线
            for i in range(map_size+1):
                # 垂直线
                cv2.line(image, (i*CELL_SIZE, 0), (i*CELL_SIZE, WINDOW_SIZE[1]), (128, 128, 128), 1)
                # 水平线
                cv2.line(image, (0, i*CELL_SIZE), (WINDOW_SIZE[0], i*CELL_SIZE), (128, 128, 128), 1)
                # 垂直线
                cv2.line(image_obs, (i*CELL_SIZE, 0), (i*CELL_SIZE, WINDOW_SIZE[1]), (128, 128, 128), 1)
                # 水平线
                cv2.line(image_obs, (0, i*CELL_SIZE), (WINDOW_SIZE[0], i*CELL_SIZE), (128, 128, 128), 1)
                # 垂直线
                cv2.line(image_all, (i*CELL_SIZE, 0), (i*CELL_SIZE, WINDOW_SIZE[1]), (128, 128, 128), 1)
                # 水平线
                cv2.line(image_all, (0, i*CELL_SIZE), (WINDOW_SIZE[0], i*CELL_SIZE), (128, 128, 128), 1)

          
            # 显示图像
            cv2.imshow("Grid Map with Unet Pathfinding", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("Grid Map with Unet Pathfinding", image_obs)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("Grid Map with Unet Pathfinding", image_answer)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            cv2.imshow("Grid Map with Unet Pathfinding", image_all)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
if __name__ == "__main__":
    train = False
    main(train)