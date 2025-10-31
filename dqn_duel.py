import numpy as np
from base import Env, BaseAgent
import torch.nn as nn
import torch
import collections
import random


def totensor(r, device=torch.device("cuda:0"), dtype=torch.float32):
    if isinstance(r, np.ndarray):
        return torch.from_numpy(r).to(device=device, dtype=dtype)
    if isinstance(r, list):
        return torch.from_numpy(np.asarray(r)).to(device=device, dtype=dtype)
    return torch.tensor([r], dtype=dtype).to(device)


def toNumpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    
class ReplayBuffer():
    
    def __init__(self, max_buffer=2000):
        self.buffer = collections.deque(maxlen=max_buffer)
        
    def add(self, s, a, r, s1, done):
        self.buffer.append((s, a, r, s1, done))
        
    def sample(self, bs=64):
        sample_data = random.sample(self.buffer, bs)
        s, a, r, s1, done = zip(*sample_data)
        return np.asarray(s), np.asarray(a), np.asarray(r), np.asarray(s1), np.asarray(done)
        
    def size(self):
        return len(self.buffer)

class DQNet(nn.Module):
    
    def __init__(self, device, env:Env, hidden_dim=16):
        super(DQNet, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.row_table = nn.Embedding(env.n_row,hidden_dim)
        self.col_table = nn.Embedding(env.n_col,hidden_dim)
        self.backbone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(2*hidden_dim, hidden_dim*2), nn.ReLU())
        self.Vnet = nn.Linear(2*hidden_dim, 1)
        self.Anet = nn.Linear(2*hidden_dim, env.n_action)

    def forward(self, r, c):
        r = totensor(r, self.device, torch.long)
        c = totensor(c, self.device, torch.long)
        x = self.row_table(r) + self.col_table(c)
        x = self.backbone(x)
        V = self.Vnet(x)
        A = self.Anet(x)
        y = V + (A - A.mean(-1).reshape(-1, 1))
        return y

class DQN(BaseAgent):
    
    def __init__(self, env):
        super().__init__(env)
        self.eps = 0.1
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.n_episode = 1000
        self.alpha = 0.1
        self.main_net = DQNet(self.device, env)
        self.target_net = DQNet(self.device, env)
        self.main_net.to(self.device)
        self.target_net.to(self.device)
        self.sync_paramters()
        self.minmal_data_size =1000
        self.replay = ReplayBuffer()
        self.critic = nn.MSELoss()
        self.optmizer = torch.optim.Adam(self.main_net.parameters(), lr=0.001)
        self.freq_update_params = 5
    
    def sync_paramters(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        
    def take_action(self, r, c):
        if np.random.uniform()<self.eps:
            a = np.random.randint(self.env.n_action)
        else:
            self.main_net.eval()
            a = self.main_net(r,c)[0].argmax().item()
            self.main_net.train()
        return a
    
    def train_qlearn(self):
        self.main_net.train()
        self.target_net.eval()
        train_time = 0
        for i_episode in range(self.n_episode):
            done = False
            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            while not done:
                a = self.take_action(r, c)
                nr, nc, reward, done = self.env.P[(r, c, a)]
                self.replay.add([r, c], a, reward, [nr, nc], done)
                if self.replay.size() > self.minmal_data_size:
                    train_time += 1
                    states, actions , rewards, next_states, dones = self.replay.sample() # s, a, r, s1, done
                    pred = self.main_net(states[:, 0], states[:, 1]).gather(1, totensor(actions, self.device, torch.long).reshape(-1, 1))
                    q1 = self.target_net(next_states[:, 0], next_states[:, 1]).max(-1)[0]
                    target = totensor(rewards, self.device) + self.gamma * q1 * totensor(1-dones)
                    loss = self.critic(pred, target.reshape(-1, 1))
                    self.optmizer.zero_grad()
                    loss.backward()
                    self.optmizer.step()
                    if train_time % self.freq_update_params==0:
                        self.sync_paramters()
                r, c = nr, nc
            self.env.visual_policy(self.get_policy())
    
    def get_policy(self):
        self.main_net.eval()
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_row)]).reshape(-1, 2)
        policy = self.main_net(index[:, 0], index[:, 1]).argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return toNumpy(policy)
    
    
def main():
    env = Env()
    agent = DQN(env)
    agent.train_qlearn()
    agent.env.visual_policy(agent.get_policy())
            
main()