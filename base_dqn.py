import numpy as np
from base import Env, BaseAgent
import torch.nn as nn
import torch
import collections
import random
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
import os
import matplotlib.pyplot as plt

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def totensor(r, dtype=torch.float32):
    if isinstance(r, np.ndarray):
        return torch.from_numpy(r).to(device=device, dtype=dtype)
    if isinstance(r, list):
        return torch.from_numpy(np.asarray(r)).to(device=device, dtype=dtype)
    if isinstance(r, torch.Tensor):
        return r.to(device)
    raise ValueError(f"类型不是我想的这三种，当前类型:{type(r)}.")


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
    
    def __init__(self, env:Env, hidden_dim=32):
        super(DQNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.row_table = nn.Embedding(env.n_row,hidden_dim)
        self.col_table = nn.Embedding(env.n_col,hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*2), 
            nn.ReLU(), 
            nn.Linear(2*hidden_dim, hidden_dim*2), 
            nn.ReLU(),
            nn.Linear(2*hidden_dim, env.n_action)
            )

    def forward(self, r, c):
        r = totensor(r, torch.long)
        c = totensor(c, torch.long)
        x = self.row_table(r) + self.col_table(c)
        y = self.net(x)
        return y

class DQN_Agent(BaseAgent):
    
    def __init__(self, env):
        super().__init__(env)
        self.n_episode = 2000
        self.minmal_data_size =1000
        self.min_eps = 0
        self.max_eps = 1.0
        self.eps = self.max_eps
        self.eps_decay = 1/1000

        self.main_net = DQNet( env).to(device)
        self.target_net = DQNet(env).to(device)
        self.sync_paramters()
        self.replay = ReplayBuffer(batch_size=128)
        self.optmizer = torch.optim.Adam(self.main_net.parameters(), lr=1e-3)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optmizer, T_max=self.n_episode, eta_min=1e-4)
        self.freq_update_params = 5

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        self.writer = SummaryWriter(log_dir=f"runs/DQN_example/{TIMESTAMP}")  # TensorBoard writer
    
    def sync_paramters(self):
        self.target_net.load_state_dict(self.main_net.state_dict())
        
    def take_action(self, r, c):
        if np.random.uniform()<self.eps:
            a = np.random.randint(self.env.n_action)
        else:
            a = self.main_net(r, c)[0].argmax().item()
        return a
    
    def train_dqn(self):
        self.main_net.train()
        self.target_net.eval()
        train_time = 0
        frame_idx = 0
        steps = []
        for i_episode in tqdm(range(self.n_episode)):
            done = False
            n_steps = 0

            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            score_1episode = 0
            loss_1episode = []
            while not done:
                frame_idx += 1
                a = self.take_action([r], [c])
                nr, nc, reward, done = self.env.P[(r, c, a)]
                self.replay.store([r, c], a, reward, [nr, nc], done)
                if self.replay.size() > self.minmal_data_size:
                    train_time += 1
                    loss = self.update_model()
                    loss_1episode.append(loss)
                    if train_time % self.freq_update_params==0:
                        self.sync_paramters()
                n_steps += 1
                score_1episode += reward
                r, c = nr, nc

            ## one episode done, update/store params/state
            steps.append(n_steps)
            self.eps = max(self.min_eps, self.eps - (self.max_eps - self.min_eps) * self.eps_decay)
            self.scheduler.step()

            avg_loss = np.mean(loss_1episode) if loss_1episode else 0
            self.writer.add_scalar("Loss/episode", avg_loss, i_episode)
            self.writer.add_scalar("Score/episode", score_1episode, i_episode)
            self.writer.add_scalar("Epsilon", self.eps, i_episode)
            self.writer.add_scalar("LearningRate", self.optmizer.param_groups[0]['lr'], i_episode)
            if i_episode%200==0:
                self.save_policy(i_episode)
        self.writer.close()
    
    def _compute_dqn_loss(self):
        trans = self.replay.sample() 
        state, next_state, reward, action, done = trans['state'], trans['next_state'], trans['reward'], trans['action'], trans['done']
        state       = totensor(state, torch.long)
        next_state  = totensor(next_state, torch.long)
        reward      = totensor(reward, torch.float32)
        action      = totensor(action, torch.long)
        done        = totensor(done, torch.float32)

        target = reward + self.gamma * self.target_net(next_state[:, 0], next_state[:, 1]).max(dim=-1)[0] * (1-done)
        pred = self.main_net(state[:, 0], state[:, 1]).gather(1, action.reshape(-1, 1))
        loss = F.smooth_l1_loss(pred, target.reshape(-1, 1))
        return loss

    def update_model(self):
        loss = self._compute_dqn_loss()
        self.optmizer.zero_grad()
        loss.backward()
        self.optmizer.step()
        return loss.item()
        

    def get_policy(self):
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_row)]).reshape(-1, 2)
        policy = self.main_net(index[:, 0], index[:, 1]).argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return tonumpy(policy)
    
    def save_policy(self, n_episode):
        arr_arrow = self.env.visual_policy(self.get_policy())
        policy_str = "\n".join(" ".join(row) for row in arr_arrow)
        self.writer.add_text("Policy", policy_str, global_step=n_episode)
    
def main():
    env = Env()
    agent = DQN_Agent(env)
    agent.train_dqn()
    agent.env.visual_policy(agent.get_policy())
            
main()