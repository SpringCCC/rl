import numpy as np
import torch
import torch.nn as nn
import random
from base import *
import torch.nn.functional as F
from springc_utils import *

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
setup_logging()

class PolicyNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.row_embed = nn.Embedding(env.n_row, hidden_dim)
        self.col_embed = nn.Embedding(env.n_col, hidden_dim)
        self.backbone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU())
        self.policy_head = nn.Sequential(nn.Linear(hidden_dim*2, env.n_action))
        
    
    def forward(self, r, c):
        r, c = totensor(r, dtype=torch.long), totensor(c, dtype=torch.long)
        x = self.row_embed(r) + self.col_embed(c)
        x = self.backbone(x)
        policy = self.policy_head(x)
        return policy
    

class ValueNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=32):
        super(ValueNet, self).__init__()
        self.row_embed = nn.Embedding(env.n_row, hidden_dim)
        self.col_embed = nn.Embedding(env.n_col, hidden_dim)
        self.backbone = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU())
        self.value_head = nn.Sequential(nn.Linear(hidden_dim*2, 1))
        
    
    def forward(self, r, c):
        r, c = totensor(r, dtype=torch.long), totensor(c, dtype=torch.long)
        x = self.row_embed(r) + self.col_embed(c)
        x = self.backbone(x)
        value = self.value_head(x)
        return value
    

class PPO(BaseAgent):
    
    def __init__(self, env:Env):
        super().__init__(env)
        self.gamma = 0.9
        self.lamb = 0.95

        self.init_net(env)
        self.n_episode = 10000
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=1e-3)
        self.max_dis = 50
        self.entropy_loss_w = 1
        self.policy_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.policy_optimizer, T_max=self.n_episode, eta_min=1e-4)
        self.value_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.value_optimizer, T_max=self.n_episode, eta_min=1e-4)
        self.n_epochs_perepisode = 5
        self.clip_eps = 0.2
    
    def init_net(self, env):
        self.policy_net = PolicyNet(env)
        self.value_net = ValueNet(env)
        self.policy_net.to(device=device)
        self.value_net.to(device=device)

    def take_action(self, r, c):
        policy = self.policy_net(r, c)
        value = self.value_net(r, c)
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
    def train_PPO_on_policy(self):
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
            v = self.value_net(r, c)
            res.append([None, None, None, None, None, v*(1-done)])
            transition_dict = self.processInfo(res)
            r, c, a, returns, advantage = transition_dict['r'], transition_dict['c'], transition_dict['actions'], transition_dict['returns'], transition_dict['advantage']
            log_prob_old_a = F.log_softmax(self.policy_net(r, c), dim=-1).gather(1, a).detach()
            for _ in range(self.n_epochs_perepisode):
                prob = self.policy_net(r, c)
                v = self.value_net(r, c)

                log_prob = F.log_softmax(prob, dim=-1)
                log_prob_a = log_prob.gather(1, a)

                ratio = torch.exp(log_prob_a - log_prob_old_a)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1-self.clip_eps, 1+self.clip_eps) * advantage
                policy_loss = torch.mean(-torch.min(surr1, surr2))
                value_loss = (0.5 * ((v-returns)**2)).mean()
                entropy_loss = (prob * log_prob).sum(dim=-1).mean()
                policy_loss = policy_loss + self.entropy_loss_w * entropy_loss

                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                policy_loss.backward()
                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
                torch.nn.utils.clip_grad_norm_(self.value_net.parameters(), 1)
                self.policy_optimizer.step()
                self.value_optimizer.step()

            self.policy_scheduler.step()
            self.value_scheduler.step()

        print("final policy:")
        self.env.visual_policy(self.get_policy())
    
    def get_policy(self):
        with torch.no_grad():
            index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_col)]).reshape(-1, 2)
            policy= self.policy_net(index[:, 0], index[:, 1])
            policy = policy.argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return toNumpy(policy)
    
                
        
def main():
    env = Env()
    agent = PPO(env)
    agent.train_PPO_on_policy()

def loop():
    for i in range(100):
        msg = f"第{i}次循环loop"
        logging.info(msg)
        main()

loop()