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
        self.backbone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim*2, env.n_action))
        self.value_head = nn.Sequential(nn.Linear(hidden_dim*2, 1))
        
    
    def forward(self, r, c):
        r, c = totensor(r, dtype=torch.long), totensor(c, dtype=torch.long)
        x = self.row_embed(r) + self.col_embed(c)
        x = self.backbone(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
    

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
        self.value_loss_w = 0.1
        self.entropy_loss_w = 1
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.n_episode, eta_min=1e-4)
        
    def take_action(self, r, c):
        policy, value = self.net(r, c)
        action_dist = torch.distributions.Categorical(logits=policy)
        a = action_dist.sample()
        return a.item(), policy, value

    def processInfo(self, res):
        advantage = 0
        returns = res[-1][-1]
        transition_dict = {'r': [], 'c':[], 'actions': [], 'returns': [], 'advantage': []}
        for i in reversed(range(len(res)-1)):
            next_value = res[i+1][-1]
            r, c, reward, a, p, v = res[i]
            delta = reward + self.gamma * next_value - v
            
            advantage = delta + self.gamma * self.lamb * advantage
            returns = reward + self.gamma * returns

            transition_dict['r'].insert(0, r)
            transition_dict['c'].insert(0, c)
            transition_dict['actions'].insert(0, a)
            transition_dict['returns'].insert(0, returns)
            transition_dict['advantage'].insert(0, advantage)
        
        for k, v in transition_dict.items():
            if k == "r"  or k=="c":
                transition_dict[k] = totensor(v).long().detach()
            elif k=='actions':
                transition_dict[k] = totensor(v).long().reshape(-1, 1).detach()
            else:
                transition_dict[k] = torch.cat(v, dim=0).to(device).detach()

        return transition_dict
    
    @sc_timing_consume()
    def train_A2C(self):
        for i_episode in range(self.n_episode):
            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            done = False
            res = []
            dis = 0
            while not done:
                a, p, v = self.take_action(r, c)
                nr, nc, reward, done = self.env.P[(r, c, a)]
                res.append([r, c, reward, a, p, v])
                r, c = nr, nc
                dis += 1
                if dis>self.max_dis:
                    break
            p, v = self.net(r, c)
            res.append([None, None, None, None, None, v*(1-done)])
            transition_dict = self.processInfo(res)
            r, c, a, returns, advantage = transition_dict['r'], transition_dict['c'], transition_dict['actions'], transition_dict['returns'], transition_dict['advantage']
            p, _ = self.net(r, c)
            probs = F.softmax(p, dim=-1)
            log_prob = F.log_softmax(p, dim=-1)
            log_prob_a = log_prob.gather(1, a)
            
            policy_loss = (- log_prob_a * advantage).sum()
            value_loss = (0.5 * ((v-returns)**2)).sum()
            entropy_loss = (probs * log_prob).sum()
            
            loss = policy_loss + self.value_loss_w * value_loss + self.entropy_loss_w * entropy_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
            self.optimizer.step()
            self.scheduler.step()
            # if i_episode % 500==0:
            #     print(f"{i_episode=}")
            #     self.env.visual_policy(self.get_policy())
        print("final policy:")
        self.env.visual_policy(self.get_policy())
    
    def get_policy(self):
        with torch.no_grad():
            index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_col)]).reshape(-1, 2)
            policy, _ = self.net(index[:, 0], index[:, 1])
            policy = policy.argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return toNumpy(policy)
    
    def get_value(self):
        with torch.no_grad():
            index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_col)]).reshape(-1, 2)
            _, v = self.net(index[:, 0], index[:, 1])
            v = v[:, 0].reshape(self.env.n_row, self.env.n_col)
        return toNumpy(v)
                
        
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