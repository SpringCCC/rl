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
from collections import deque

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def totensor(r, dtype=torch.float32):
    if isinstance(r, np.ndarray):
        return torch.from_numpy(r).to(device=device, dtype=dtype, non_blocking=True)
    if isinstance(r, list):
        return torch.from_numpy(np.asarray(r)).to(device=device, dtype=dtype, non_blocking=True)
    if isinstance(r, torch.Tensor):
        return r.to(device, non_blocking=True)
    raise ValueError(f"类型不是我想的这三种，当前类型:{type(r)}.")


def tonumpy(a):
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    
class ReplayBuffer():
    
    def __init__(self, state_dim=2, max_buffer=2000, batch_size=32, n_step=3, gamma=0.9):
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
        self.n_step = n_step
        self.n_step_buffer = deque(maxlen=self.n_step)
        self.gamma = gamma
        
    
    def _merge_n_step_info(self):
        s, a, R, s1, done = self.n_step_buffer[-1]
        for i in reversed(range(self.n_step-1)):
            r = self.n_step_buffer[i][2]
            R = r + self.gamma * R
        return R

        

    def store(self, trans): #trans:s, a, r, s1, done

        self.n_step_buffer.append(trans)

        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        s, a, r, s1, done           = self.n_step_buffer[0]
        s_t, a_t, r_t, s1_t, done_t = self.n_step_buffer[-1]

        R = self._merge_n_step_info()

        self.state[self.cur_ptr] = np.asarray(s)
        self.next_state[self.cur_ptr] = np.asarray(s1_t)
        self.reward[self.cur_ptr] = R
        self.action[self.cur_ptr] = a
        self.done[self.cur_ptr] = done_t
        self.cur_size = min(self.max_size, self.cur_size+1)
        self.cur_ptr = (self.cur_ptr+1)%self.max_size

        if done_t:
            self.n_step_buffer.clear()
        return s, a, r, s1, done
        
    def sample(self, idx=None):
        if idx is None:
            # idx = np.random.choice(self.cur_size, self.batch_size, replace=False)
            idx = np.random.randint(0, self.cur_size, size=self.batch_size)
        trans = {}
        trans['state'] = self.state[idx]
        trans['next_state'] = self.next_state[idx]
        trans['reward'] = self.reward[idx]
        trans['action'] = self.action[idx]
        trans['done'] = self.done[idx]
        trans['idx'] = idx
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
            
            )
        self.v_head = nn.Linear(2*hidden_dim, 1)
        self.a_head = nn.Linear(2*hidden_dim, env.n_action)

    def forward(self, r, c):
        r = totensor(r, torch.long)
        c = totensor(c, torch.long)
        x = self.row_table(r) + self.col_table(c)
        x = self.net(x)
        v = self.v_head(x)
        a = self.a_head(x)
        y = v + a - a.mean(dim=-1, keepdim=True)
        return y

class DQN_Agent(BaseAgent):
    
    def __init__(self, env):
        super().__init__(env)
        self.n_episode = 2000
        self.minmal_data_size =1000
        self.min_eps = 0.1
        self.max_eps = 1.0
        self.eps = self.max_eps
        self.eps_decay = 1/1000

        self.main_net = DQNet( env).to(device)
        self.target_net = DQNet(env).to(device)
        self.sync_paramters()
        self.replay_1 = ReplayBuffer(batch_size=128, n_step=1, gamma=self.gamma)
        self.replay_n = ReplayBuffer(batch_size=128, n_step=3, gamma=self.gamma)

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
            with torch.no_grad():
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
                trans = ([r, c], a, reward, [nr, nc], done)
                one_trans = self.replay_n.store(trans)
                if one_trans:
                    self.replay_1.store(one_trans)


                if self.replay_1.size() > self.minmal_data_size:
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
            if i_episode > (self.n_episode -100):
                self.eps = 0
            self.scheduler.step()

            avg_loss = np.mean(loss_1episode) if loss_1episode else 0
            self.writer.add_scalar("Loss/episode", avg_loss, i_episode)
            self.writer.add_scalar("Score/episode", score_1episode, i_episode)
            self.writer.add_scalar("Epsilon", self.eps, i_episode)
            self.writer.add_scalar("LearningRate", self.optmizer.param_groups[0]['lr'], i_episode)
            # if i_episode%200==0:
            #     self.save_policy(i_episode)
            print(f"第{i_episode}条epissode")
            self.save_policy(i_episode)
        self.writer.close()
    
    def _compute_dqn_loss(self, trans, gamma):

        state, next_state, reward, action, done = trans['state'], trans['next_state'], trans['reward'], trans['action'], trans['done']
        state       = totensor(state, torch.long)
        next_state  = totensor(next_state, torch.long)
        reward      = totensor(reward, torch.float32)
        action      = totensor(action, torch.long)
        done        = totensor(done, torch.float32)
        with torch.no_grad():
            target_action = self.main_net(next_state[:, 0], next_state[:, 1]).argmax(dim=1, keepdim=True)
            next_q_value = self.target_net(next_state[:, 0], next_state[:, 1]).gather(1, target_action).detach()
            target = reward + gamma * next_q_value[:, 0] * (1-done)
        
        pred = self.main_net(state[:, 0], state[:, 1]).gather(1, action.reshape(-1, 1))
        loss = F.smooth_l1_loss(pred, target.reshape(-1, 1))
        return loss

    def update_model(self):
        samples = self.replay_1.sample()
        loss_1 = self._compute_dqn_loss(samples, self.gamma)
        samples = self.replay_n.sample(samples['idx'])
        loss_n = self._compute_dqn_loss(samples, self.gamma**self.replay_n.n_step)
        loss = loss_1 + loss_n
        self.optmizer.zero_grad()
        loss.backward()
        self.optmizer.step()
        return loss.item()
        

    def get_policy(self):
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_col)]).reshape(-1, 2)
        policy = self.main_net(index[:, 0], index[:, 1]).argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return tonumpy(policy)
    
    def save_policy(self, n_episode):
        arr_arrow = self.env.visual_policy(self.get_policy())
        policy_str = "\n\n".join(" ".join(row) for row in arr_arrow)
        self.writer.add_text("Policy", policy_str, global_step=n_episode)
    
def main():
    env = Env()
    agent = DQN_Agent(env)
    agent.train_dqn()
    agent.env.visual_policy(agent.get_policy())
            
main()