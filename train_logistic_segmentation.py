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
class PathDataset(Dataset):
    def __init__(self, X, Y):
        # 转换为PyTorch张量并添加通道维度
        self.X = torch.FloatTensor(np.array(X)).unsqueeze(1)  # (N, 1, 25, 25)
        self.Y = torch.FloatTensor(np.array(Y)).unsqueeze(1)  # (N, 1, 25, 25)
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# 训练函数
def train_model(X, Y, X_test, Y_test,number_epochs=100,lr=0.01):
    # 数据准备
    train_dataloader = DataLoader(PathDataset(X, Y), batch_size=512, shuffle=True)
    test_dataloader = DataLoader(PathDataset(X_test, Y_test), batch_size=3, shuffle=True)

    # 模型初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet().to(device)
    
    
    # 损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()  # 二值交叉熵损失
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 训练循环
    for epoch in range(number_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_Y in train_dataloader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            # 前向传播
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 打印统计信息
        train_avg_loss = epoch_loss / len(train_dataloader)
        train_iou = calculate_iou(outputs, batch_Y)

        model.eval()
        epoch_loss = 0.0
        
        for batch_X, batch_Y in test_dataloader:
            batch_X = batch_X.to(device)
            batch_Y = batch_Y.to(device)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
    
            epoch_loss += loss.item()
        
        # 打印统计信息
        test_avg_loss = epoch_loss / len(test_dataloader)
        test_iou = calculate_iou(outputs, batch_Y)
        
        print(f'Epoch [{epoch}/{number_epochs}],Train Loss: {train_avg_loss:.6f},Train IOU:{train_iou:.6f},Test Loss: {test_avg_loss:.6f},Test IOU:{test_iou:.6f}')
        if test_iou == 1.0:
            return model
    return model

def predict_path(X, Y, model,grid_envs,repeat):
    # 数据准备
    dataset = PathDataset(X, Y)
    dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    
    for batch_X, batch_Y in dataloader:
        batch_X = batch_X.to(device)
        batch_Y = batch_Y.to(device)
        start_time = time.time()
        for i in range(repeat):
            outputs = model(batch_X)
        end_time = time.time()
        final_results = []
        for i in range(0,3):
            case = outputs[i,0,:,:].cpu().detach().numpy()
            grid_env = grid_envs[i]
            for y in range(len(case)):
                    for x in range(len(case[0])):
                        color_float = case[x][y]
                        if color_float>0.5:
                             case[y][x] = 4
                             grid_env[y][x] = 4
                        else:
                             case[y][x] = 0
            final_results.append(grid_env)

        return final_results,end_time-start_time
        

def calculate_iou(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = (pred + target).clamp(0,1).sum()
    return (intersection + 1e-6) / (union + 1e-6)

def weights_init(m):
    """自定义初始化函数"""
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # 卷积层和转置卷积层初始化
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in', nonlinearity='relu')
        
        # 偏置项初始化为0
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    
    elif isinstance(m, nn.BatchNorm2d):
        # 批归一化层初始化
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) × 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """下采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
            nn.Dropout(p=0.1)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """上采样模块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # 使用转置卷积进行上采样
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 上采样并处理尺寸差异
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        
        # 下采样路径
        self.inc = DoubleConv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512) 
        self.down4 = Down(512, 1024) 
        
        # 上采样路径
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)  
        
        # 输出层
        self.outc = nn.Conv2d(64, 1, kernel_size=1)
        self.apply(weights_init) 

    def forward(self, x):
        # 编码器
        x1 = self.inc(x)       
        x2 = self.down1(x1)    
        x3 = self.down2(x2)    
        x4 = self.down3(x3)  
        x5 = self.down4(x4)    
        
        # 解码器
        x = self.up1(x5, x4)  
        x = self.up2(x, x3)   
        x = self.up3(x, x2)  
        x = self.up4(x, x1)    
        x = self.outc(x)       
        return x  # 直接输出logits
    
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
def main(train,num_maze):
    if train:
        X = []
        Y = []
        X_test = []
        Y_test = []
        while len(X)<num_maze:
            start = (25, 1)
            goal = (1, 25)


            # 初始化参数（与原始代码一致）
            map_size = 25
            grid = np.zeros((map_size, map_size))
            grid_answer = np.zeros((map_size, map_size))

            CELL_SIZE = 30
            WINDOW_SIZE = (map_size*CELL_SIZE, map_size*CELL_SIZE)
            
            # 创建空白图像
            image = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            image_answer = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            
            # 颜色定义 (BGR格式)
            COLORS = {
                0: (255, 255, 255),    # 空地
                1: (0, 0, 0),          # 障碍物
                2: (0, 255, 0),        # 起点 (绿色)
                3: (0, 255, 255),      # 路径 (黄色)
                4: (0, 0, 255)         # 终点 (红色)
            }
            
            # 数据预处理
            def prepare_data(start,goal):
                ratio = random.randint(0, 5)/100
                obstacle_cells = int((map_size* map_size) * ratio)  # 障碍物占地图的0-5%
                obs = []
                obstacles = set()
                for i in range(obstacle_cells):
                    x = random.randint(1, map_size)
                    y = random.randint(1, map_size)
                    while (x == start[0] and y == start[1]) or (x == goal[0] and y == goal[1]) or (x,y) in obs:
                    # 判断是否为合法障碍物
                        x = random.randint(1, map_size)
                        y = random.randint(1, map_size)
                    obs.append((x, y))

                for (x, y) in obs:
                    grid_x, grid_y = x-1, y-1
                    grid[grid_x][grid_y] = 1  
                    obstacles.add((grid_x, grid_y))
                
                # 设置起点终点
                start_x, start_y = start[0]-1, start[1]-1
                grid[start_y][start_x] = 2
                goal_x, goal_y = goal[0]-1, goal[1]-1
                grid[goal_y][goal_x] = 3
                
                return obstacles
            
            obstacles = prepare_data(start,goal)
            
            # 运行A*算法寻找路径
            path = a_star(start, goal, obstacles)
            
            if path:
                for (x, y) in path:
                    grid_answer[x,y]=3
                X.append(grid)
                Y.append(grid_answer)


        for case_num in range(1,4):
            start = (25, 1)
            goal = (1, 25)


            # 初始化参数（与原始代码一致）
            map_size = 25
            grid = np.zeros((map_size, map_size))
            grid_answer = np.zeros((map_size, map_size))

            CELL_SIZE = 30
            WINDOW_SIZE = (map_size*CELL_SIZE, map_size*CELL_SIZE)
            
            # 创建空白图像
            image = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            image_answer = np.ones((WINDOW_SIZE[1], WINDOW_SIZE[0], 3), dtype=np.uint8) * 255
            
            # 颜色定义 (BGR格式)
            COLORS = {
                0: (255, 255, 255),    # 空地
                1: (0, 0, 0),          # 障碍物
                2: (0, 255, 0),        # 起点 (绿色)
                3: (0, 255, 255),      # 终点 (黄色)
                4: (255, 0, 0)         # 路径 (蓝色)
            }
            
            # 数据预处理
            def prepare_data(case_num):
                # 填充障碍物
                obstacles = set()
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
                    obstacles.add((grid_x, grid_y))
                
                # 设置起点终点
                start_x, start_y = start[0]-1, start[1]-1
                grid[start_y][start_x] = 2
                goal_x, goal_y = goal[0]-1, goal[1]-1
                grid[goal_y][goal_x] = 3
                
                return obstacles
            
            obstacles = prepare_data(case_num)
            
            # 运行A*算法寻找路径
            path = a_star(start, goal, obstacles)
            
            if path:
                for (x, y) in path:
                    grid_answer[x,y]=3
                X_test.append(grid)
                Y_test.append(grid_answer)
        X = np.array(X) # include obs, start, and goal
        Y = np.array(Y)/3 # scale to 0-1, where 1 is the right path
        X_test = np.array(X_test) # include obs, start, and goal
        Y_test = np.array(Y_test)/3 # scale to 0-1, where 1 is the right path
    
        print(f"shape of training dataset {np.shape(X)}, {np.shape(Y)}")
        print(f"shape of testing dataset {np.shape(X_test)}, {np.shape(Y_test)}")

        # print("example\n",X[0],Y[0])
        start_time = time.time()
        trained_model = train_model(X,Y,X_test,Y_test,number_epochs=100,lr=0.01)
        print("Training time: ",time.time()-start_time)
        torch.save(trained_model.state_dict(), 'trained_model.pth')

    else:
        pass
if __name__ == "__main__":
    train = True
    num_maze = 30000
    main(train,num_maze)