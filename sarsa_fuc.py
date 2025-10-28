
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from springc_utils import *

class SCNet(nn.Module):

    def __init__(self,x_size=5, y_size=5, a_size=4, embed_dim=32):
        super(SCNet, self).__init__()
        self.x_emb = nn.Embedding(x_size, embed_dim)
        self.y_emb = nn.Embedding(y_size, embed_dim)
        self.a_emb = nn.Embedding(a_size, embed_dim)
        self.net0 = nn.Sequential(nn.Linear(2*embed_dim, 4*embed_dim), nn.ReLU(), nn.Linear(4*embed_dim, 2*embed_dim), nn.ReLU(), nn.Linear(2*embed_dim, 4))
    

    def forward(self, x0, y0, a0):
        x0 = toTensor(x0)
        y0 = toTensor(y0)
        a0 = toTensor(a0)
        x = self.x_emb(x0)
        y = self.x_emb(y0)
        a = self.x_emb(a0)
        input_v = torch.cat([x+y, a], dim=-1)
        out = self.net0(input_v)
        return out


class Env():

    def __init__(self, gridSize, dst_pos, zhangai_pos):
        self.step_reward = -1
        self.dst_reward = 10
        self.zhangai_reward = -2
        self.gridSize = gridSize
        self.m = gridSize[0] # row
        self.n = gridSize[1] # cool
        self.grid = np.zeros(gridSize)
        self.n_actions = 4
        self.step_reward = -1
        self.mv = {
            0: [-1, 0],#上 x,y
            1: [1, 0],#下
            2: [0, -1],#左
            3: [0, 1],#右
        }
        self.model = SCNet()
        self.dst_pos = dst_pos
        self.zhangai_pos = zhangai_pos
        self.P = self.init_P()
        self.gamma = 0.9
        self.delta_thresh = 0.0001
        self.n_eplisode = 1000
        self.eps = 0.1
        self.alpha = 0.1
        self.criterion = nn.MSELoss()   
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)


    def check_terminal(self, pos):
        if pos in self.dst_pos:
            return True
        else:
            return False
    

    def init_P(self):
        P = {}
        for i in range(self.m): 
            for j in range(self.n):
                for k in range(self.n_actions):
                    next_i = i + self.mv[k][0]
                    next_j = j + self.mv[k][1]
                    if not (0 <= next_i < self.m and 0 <= next_j < self.n):
                        P[(i, j, k)] = (i, j, self.step_reward)
                    else:
                        if [next_i, next_j] in self.dst_pos:
                            P[(i, j, k)] = (next_i, next_j, self.dst_reward)
                        elif [next_i, next_j] in self.zhangai_pos:
                            P[(i, j, k)] = (i, j, self.zhangai_reward)
                        else:
                            P[(i, j, k)] = (next_i, next_j, self.step_reward)
        return P
    
    def gen_action(self, i, j):
        if np.random.uniform() < self.eps:#小概率
            a = np.random.randint(0, self.n_actions)
        else:
            a = self.model(i, j).argmax()
        return a



    def gen_v(self):
        for i in range(self.m):
            for j in range(self.n):
                self.v[i, j] = self.q_table[i, j].max()
        return self.v
    
    



    def visualize_policy(self):
        best_policy = self.q_table.argmax(-1)
        grid_display = np.full((self.m, self.n), '', dtype=object)
        arrow_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}

        for i in range(self.m):
            for j in range(self.n):
                if [i,j] in self.dst_pos:
                    grid_display[i,j] = '★'
                elif [i,j] in self.zhangai_pos:
                    grid_display[i,j] = '■'
                else:
                    grid_display[i,j] = arrow_dict[best_policy[i,j]]

        print("Policy Grid (★目标, ■障碍):")
        for row in grid_display:
            print(' '.join(row))
                


    def sarsa_iter(self):
        for t in range(self.n_eplisode):
            print(f"第{t}条轨迹")
            self.model.train()
            i, j = 0, 0
            while True:
                i, j = np.random.randint(0, self.n), np.random.randint(0, self.n)
                if self.check_terminal([i, j]):
                    continue
                else:
                    break
            while True:
                a = self.gen_action(i, j)
                next_i, next_j, r = self.P[(i, j, a)]
                if self.check_terminal([next_i, next_j]):
                    self.q_table[i, j, a] += self.alpha *(r - self.q_table[i, j, a])
                    break
                next_a = self.gen_action(next_i, next_j)

                pred = r + self.gamma * self.model(next_i, next_j, next_a)
                target = self.model(i, j, a)

                self.optimizer.zero_grad()
                loss = self.criterion(pred, target)
                loss.backward()        # 计算梯度
                self.optimizer.step()       # 参数更新


        # 迭代完成
        self.model.eval()
        best_policy = np.zeros((5,5), dtype=np.int32)
        for i in range(5):
            for j in range(5):
                v = []
                for k in range(4):
                    v.append(self.model(i, j, k).item())
                best_policy[i, j] = np.asarray(v).argmax()


        arrow_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arr_arrow = np.vectorize(arrow_dict.get)(best_policy)
        print(f"best_policy:\n")
        print(arr_arrow)
        self.visualize_policy()

            
def main():
    dst = [[3, 2]]
    zhangai = [[1,1], [1, 2], [2,2], [3,1], [3, 3], [4, 1]]
    gridSize = [5, 5]
    env = Env(gridSize, dst, zhangai)
    env.sarsa_iter()
    
                    
main()