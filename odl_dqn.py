import numpy as np
from base import Env, BaseAgent
import torch.nn as nn
import torch
import collections
import random
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def totensor(r, dtype=torch.float32):
    if isinstance(r, np.ndarray):
        return torch.from_numpy(r).to(device=device, dtype=dtype)
    if isinstance(r, list):
        return torch.from_numpy(np.asarray(r)).to(device=device, dtype=dtype)
    return torch.tensor([r], dtype=dtype).to(device)


def tonumpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()


class ReplayBuffer():
    
    def __init__(self, state_dim=2, max_buffer=2000, batch_size=32):
        self.buffer = collections.deque(maxlen=max_buffer)
        self.state = np.zeros([max_buffer, state_dim], dtype=np.int32)
        self.next_state = np.zeros([max_buffer, state_dim], dtype=np.int32)
        self.reward = np.zeros([max_buffer], dtype=np.float32)
        self.action = np.zeros([max_buffer], dtype=np.int32)
        self.done = np.zeros([max_buffer], dtype=np.float32)
        self.max_size = max_buffer
        self.cur_ptr = 0
        self.cur_size = 0
        self.batch_size = batch_size
        
        
    def store(self, s, a, r, s1, done):
        self.state[self.cur_ptr] = np.asarray(s)
        self.next_state[self.cur_ptr] = np.asarray(s1)
        self.reward[self.cur_ptr] = r
        self.action[self.cur_ptr] = a
        self.done[self.cur_ptr] = done
        self.cur_size = min(self.max_size, self.cur_size+1)
        self.cur_ptr = (self.cur_ptr+1)%self.max_size

        
    def sample(self, bs=64):
        idx = np.random.choice(self.cur_size, self.batch_size, replace=False)
        trans = {}
        trans['state'] = self.state[idx]
        trans['next_state'] = self.next_state[idx]
        trans['reward'] = self.reward[idx]
        trans['action'] = self.action[idx]
        trans['done'] = self.done[idx]
        return trans
        
    def size(self):
        return self.cur_size



class DQNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=16):
        super(DQNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.row_table = nn.Embedding(env.n_row,hidden_dim)
        self.col_table = nn.Embedding(env.n_col,hidden_dim)
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(2*hidden_dim, hidden_dim*2), nn.ReLU(),nn.Linear(2*hidden_dim, env.n_action))

    def forward(self, r, c):
        r = totensor(r, torch.long)
        c = totensor(c, torch.long)
        x = self.row_table(r) + self.col_table(c)
        y = self.net(x)
        return y

class DQN(BaseAgent):
    
    def __init__(self, env):
        super().__init__(env)
        self.n_episode = 1000
        self.main_net = DQNet(env)
        self.target_net = DQNet(env)
        self.main_net.to(device)
        self.target_net.to(device)
        self.sync_paramters()

        self.minmal_data_size =1000
        self.replay = ReplayBuffer(max_buffer=2000, batch_size=128)
        self.critic = nn.SmoothL1Loss()
        # self.critic = nn.MSELoss()
        self.optmizer = torch.optim.Adam(self.main_net.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optmizer, T_max=self.n_episode, eta_min=1e-4)

        self.freq_update_params = 5

        self.max_eps = 1.0
        self.eps = self.max_eps
        self.min_eps = 0.1
        self.eps_decay = 1/200
    
    def sync_paramters(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
        
        
    def take_action(self, r, c):
        if np.random.uniform()<self.eps:
            a = np.random.randint(self.env.n_action)
        else:
            a = self.main_net(r,c)[0].argmax().item()
        return a
    
    def train_qlearn(self):
        self.main_net.train()
        self.target_net.eval()
        train_time = 0
        for i_episode in range(self.n_episode):
            done = False
            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            print(f"第{i_episode}条episdode, {self.eps = }")
            while not done:
                a = self.take_action(r, c)
                nr, nc, reward, done = self.env.P[(r, c, a)]
                self.replay.store([r, c], a, reward, [nr, nc], done)
                if self.replay.size() > self.minmal_data_size:
                    train_time += 1
                    # states, actions , rewards, next_states, dones = self.replay.sample()
                    trans = self.replay.sample()
                    states = trans['state']
                    actions = trans['action']
                    rewards = trans['reward']
                    next_states = trans['next_state']
                    dones = trans['done']
                    
                     # s, a, r, s1, done
                    pred = self.main_net(states[:, 0], states[:, 1]).gather(1, totensor(actions, torch.long).reshape(-1, 1))
                    q1 = self.target_net(next_states[:, 0], next_states[:, 1]).max(-1)[0]
                    target = totensor(rewards) + self.gamma * q1 * totensor(1-dones)
                    loss = self.critic(pred, target.reshape(-1, 1))
                    self.optmizer.zero_grad()
                    loss.backward()
                    self.optmizer.step()
                    if train_time % self.freq_update_params==0:
                        self.sync_paramters()
                r, c = nr, nc
            self.eps = max(self.min_eps, self.eps - (self.max_eps-self.min_eps)*self.eps_decay)
    

    def get_policy(self):
        self.main_net.eval()
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_row)]).reshape(-1, 2)
        policy = self.main_net(index[:, 0], index[:, 1]).argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return tonumpy(policy)
    
    
def main():
    env = Env()
    agent = DQN(env)
    agent.train_qlearn()
    agent.env.visual_policy(agent.get_policy())
            
main()