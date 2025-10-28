import numpy as np
import torch
import torch.nn as nn
import collections
import random


import torch
import torch.nn as nn
import torch.optim as optim


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

class QNet(nn.Module):
    
    def __init__(self, n_x, n_y, n_action, device, hidden_dim=32):
        super(QNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim),nn.ReLU(), nn.Linear(hidden_dim, n_action))
        self.x_embed = nn.Embedding(n_x, hidden_dim)
        self.y_embed = nn.Embedding(n_y, hidden_dim)
        self.device = device
        
        
    def forward(self, i, j):
        x = self.x_embed(i)
        y = self.y_embed(j)
        input_v = x + y
        out = self.net(input_v)
        return out

class DQN():
    
    def __init__(self):
        self.eps = 0.1
        self.n_action = 4
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.param_model = QNet(5, 5, self.n_action, self.device).to(self.device)
        self.target_model = QNet(5, 5, self.n_action, self.device).to(self.device)
        self.target_model.eval()
        self.criterm = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.param_model.parameters(), lr=0.001)
        self.gamma = 0.9
        self.target_update_freq = 10
        self.update_count = 0
        
    def copy_params(self):
        self.target_model.load_state_dict(self.param_model.state_dict())
    
    def tack_action(self, i, j):
        if np.random.uniform()<self.eps:
            a = np.random.randint(self.n_action)
        else:
            with torch.no_grad():
                a = self.param_model(torch.tensor([i]).to(self.device), torch.tensor([j]).to(self.device)).argmax(dim=-1).item()
        return a
    
    def update(self, b_s, b_a, b_r, b_ns, b_d):
        b_s = torch.tensor(b_s, dtype=torch.long).to(self.device)
        b_a = torch.tensor(b_a, dtype=torch.long).to(self.device)
        b_r = torch.tensor(b_r, dtype=torch.float32).to(self.device)
        b_ns = torch.tensor(b_ns, dtype=torch.long).to(self.device)
        b_d = torch.tensor(b_d, dtype=torch.float32).to(self.device)
        state_r = b_s[:, 0]
        state_c = b_s[:, 1]
        next_state_r = b_ns[:, 0]
        next_state_c = b_ns[:, 1]
        
        pred = self.param_model(state_r, state_c).gather(1, b_a.reshape(-1, 1))
        target = b_r + self.gamma * torch.max(self.target_model(next_state_r, next_state_c), dim=-1)[0] * (1-b_d)
        loss = self.criterm(pred, target[:,None])
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.copy_params()
            
        
    
class ReplayBuffer():
    
    def __init__(self, capacity=2000):
        self.buffer = collections.deque(maxlen=capacity)
        
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return (
            np.array(state),
            np.array(action),
            np.array(reward, dtype=np.float32),
            np.array(next_state),
            np.array(done, dtype=np.float32)
        )
    
    def size(self):
        return len(self.buffer)
    
    
    
def train():
    agent = DQN()
    myenv = Env()
    n_eplisode = 100
    buffer = ReplayBuffer()
    minimal_size = 500
    batch_size = 64
    best_policy = np.zeros((5, 5))
    
    for i in range(n_eplisode):
        print(f"第{i}条轨迹")
        done = False
        myenv.reset()
        steps = 0
        while not done:
            steps += 1
            action = agent.tack_action(myenv.r, myenv.c)
            next_r, next_c, r, done = myenv.step(myenv.r, myenv.c, action)
            buffer.add([myenv.r, myenv.c], action, r, [next_r, next_c], done)
            myenv.r, myenv.c = next_r, next_c
            if buffer.size() > minimal_size:
                # print("train")
                b_s, b_a, b_r, b_ns, b_d = buffer.sample(batch_size)
                agent.update(b_s, b_a, b_r, b_ns, b_d)
                
                
        with torch.no_grad():        
            for ai in range(5):
                for aj in range(5):
                    best_policy[ai, aj] = agent.param_model(torch.tensor([ai]).to(agent.device), torch.tensor([aj]).to(agent.device)).argmax(dim=-1).item()
                
                
        arrow_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arr_arrow = np.vectorize(arrow_dict.get)(best_policy)
        print(f"best_policy:\n")
        print(arr_arrow)

            
            
train()
            
            
            
            