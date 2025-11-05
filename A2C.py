import numpy as np
import torch
import torch.nn as nn
import random
from base import *
import torch.nn.functional as F
from springc_utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
setup_logging()

class A2CNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=32):
        super(A2CNet, self).__init__()
        self.row_embed = nn.Embedding(env.n_row, hidden_dim)
        self.col_embed = nn.Embedding(env.n_col, hidden_dim)
        self.backbone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(hidden_dim*2, hidden_dim*2), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim*2, env.n_action))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim*2, 1))
        
    
    def forward(self, r, c):
        r, c = totensor(r, dtype=torch.long), totensor(c, dtype=torch.long)
        x = self.row_embed(r) + self.col_embed(c)
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    
    
@sc_timing_consume()
class A2C(BaseAgent):
    
    def __init__(self, env:Env):
        super().__init__(env)
        self.gamma = 0.9
        self.lamb = 0.95
        self.net = A2CNet(env)
        self.net.to(device=device)
        self.n_episode = 10000
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        self.max_dis = 50
        self.value_loss_w = 0.5
        self.entropy_loss_w = 0.1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_episode, eta_min=1e-4)
        
    def take_action(self, r, c):
        # self.net.eval()
        policy, value = self.net(r, c)
        action_dist = torch.distributions.Categorical(logits=policy)
        a = action_dist.sample()
        # self.net.train()
        return a.item()
    
    def processInfo(self, res):
        out = [None] * (len(res) - 1)
        advantage = 0
        returns = res[-1][-1]
        actions = [None] * (len(res) - 1)
        for i in reversed(range(len(res)-1)):
            next_value = res[i+1][-1]
            reward, a, p, v = res[i]
            delta = reward + self.gamma * next_value - v
            advantage = delta + self.gamma * self.lamb * advantage
            returns = reward + self.gamma * returns
            out[i] = p, v, returns, advantage
            actions[i] =a
        return map(lambda x:torch.cat(x, dim=0), zip(*out)), actions
    
    def train_A2C(self):
        for i_episode in range(self.n_episode):
            self.env.reset()
            self.optimizer.zero_grad()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            done = False
            res = []
            dis = 0
            self.net.train()
            while not done:
                a = self.take_action(r, c)
                nr, nc, reward, done = self.env.P[(r, c, a)]
                p, v = self.net(r, c)
                res.append([reward, a, p, v])
                r, c = nr, nc
                dis += 1
                if dis>self.max_dis:
                    break
            p, v = self.net(r, c)
            res.append([None, None, None, v*(1-done)])
            (p, v, returns, advantage), a = self.processInfo(res)
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            
            a = totensor(a, dtype=torch.long)
            probs = F.softmax(p, dim=-1)
            log_prob = F.log_softmax(p, dim=-1)
            log_prob_a = log_prob.gather(1, a.detach().reshape(-1, 1))
            
            policy_loss = (- log_prob_a * advantage.detach()).sum()
            value_loss = (0.5 * ((v-returns.detach())**2)).sum()
            entropy_loss = (probs * log_prob).sum()
            loss = policy_loss + self.value_loss_w * value_loss + self.entropy_loss_w * entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()
            # if i_episode % 50==0:
            #     print(f"{i_episode=}")
            #     self.env.visual_policy(self.get_policy())
        # print("final policy:")
        self.env.visual_policy(self.get_policy())
    
    def get_policy(self):
        self.net.eval()
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_col)]).reshape(-1, 2)
        policy, _ = self.net(index[:, 0], index[:, 1])
        policy = policy.argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return toNumpy(policy)
                
                
        
def main():
    env = Env()
    agent = A2C(env)
    agent.train_A2C()


def loop():
    for i in range(100):
        msg = f"第{i}次循环loop"
        logging.info(msg)
        main()

loop()