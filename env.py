import torch
import torch.nn as nn
import torch.optim as optim
from springc_utils import *


class Env():
    
    def __init__(self, row=5, col=5):
        self.dst_pos = [[3, 2]]
        self.zhangai_pos = [[1,1], [1, 2], [2,2], [3,1], [3, 3], [4, 1]]
        self.step_reward = -1
        self.n_actions = 4
        self.zhangai_reward = -5
        self.dst_reward = 10
        self.mv = {
            0: [-1, 0],#上 x,y
            1: [1, 0],#下
            2: [0, -1],#左
            3: [0, 1],#右
        }
        self.row = row
        self.col = col
        self.r = 0
        self.c = 0
        self.P = self.init_P()
        self.reset()
        
    def check_done(self, p):
        return True if p in self.dst_pos else False
    
    def init_P(self):
        P = {}
        for i in range(self.row): 
            for j in range(self.col):
                for k in range(self.n_actions):
                    next_i = i + self.mv[k][0]
                    next_j = j + self.mv[k][1]
                    if not (0 <= next_i < self.row and 0 <= next_j < self.col):
                        P[(i, j, k)] = (i, j, self.step_reward)
                    else:
                        if [next_i, next_j] in self.dst_pos:
                            P[(i, j, k)] = (next_i, next_j, self.dst_reward)
                        elif [next_i, next_j] in self.zhangai_pos:
                            P[(i, j, k)] = (i, j, self.zhangai_reward)
                        else:
                            P[(i, j, k)] = (next_i, next_j, self.step_reward)
        return P
    
    
    def step(self, i, j, a):
        next_i, next_j, r = self.P[(i, j, a)]
        done = self.check_done([next_i, next_j])
        return next_i, next_j, r, done
        
    
    def reset(self):
        while True:
            self.r = np.random.randint(self.row)
            self.c = np.random.randint(self.col)
            if self.check_done([self.r, self.c]):
                continue
            else:
                break