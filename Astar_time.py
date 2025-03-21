import numpy as np
import heapq  # 用于优先队列
import cv2
import math
import random
import time 
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

# 主程序中添加路径规划和绘制
def main(num_maze,start,goal,maze=[]):

    while len(maze)<num_maze:
        # 初始化参数（与原始代码一致）
        map_size = 25
        
        # 数据预处理
        def prepare_data(start,goal):
            ratio = random.randint(0, 30)/100
            obstacle_cells = int((map_size* map_size) * ratio)  # 障碍物占据40%的格子
            obs = []
            obstacles = set()
            for i in range(obstacle_cells):
                x = random.randint(1, map_size)
                y = random.randint(1, map_size)
                while (x == start[0] and y == start[1]) or (x == goal[0] and y == goal[1]) or (x,y) in obs:
                    x = random.randint(1, map_size)
                    y = random.randint(1, map_size)
                obs.append((x, y))

            for (x, y) in obs:
                grid_x, grid_y = x-1, y-1 
                obstacles.add((grid_x, grid_y))
            
            return obstacles
        maze.append(prepare_data(start,goal))
    return maze 
            
if __name__ == "__main__":
    num_maze = 2048
    start_time = time.time()
    start = (25, 1)
    goal = (1, 25)
    obstacles = main(num_maze,start,goal)
    for _ in range(10):
        # 运行A*算法寻找路径
        path = a_star(start, goal, obstacles)
    print(f"solving time: {(time.time()-start_time)/num_maze/10}")