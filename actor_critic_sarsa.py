import numpy as np
import torch
import torch.nn as nn
import random
from base import *
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

class QNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=32):
        super(QNet, self).__init__()
        self.row_embed = nn.Embedding(env.n_row, hidden_dim)
        self.col_embed = nn.Embedding(env.n_col, hidden_dim)
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(2*hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(2*hidden_dim, env.n_action))
        
        nn.init.xavier_uniform_(self.row_embed.weight)
        nn.init.xavier_uniform_(self.col_embed.weight)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, r, c):
        r, c = totensor(r, dtype=torch.long), totensor(c, dtype=torch.long)
        x = self.row_embed(r) + self.col_embed(c)
        y = self.net(x)
        return y

class PolicyNet(nn.Module):
    
    def __init__(self, env:Env, hidden_dim=32):
        super(PolicyNet, self).__init__()
        self.row_embed = nn.Embedding(env.n_row, hidden_dim)
        self.col_embed = nn.Embedding(env.n_col, hidden_dim)
        self.net = nn.Sequential(nn.Linear(hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(2*hidden_dim, hidden_dim*2), nn.ReLU(), nn.Linear(2*hidden_dim, env.n_action))
        
        nn.init.xavier_uniform_(self.row_embed.weight)
        nn.init.xavier_uniform_(self.col_embed.weight)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, r, c):
        r, c = totensor(r, dtype=torch.long), totensor(c, dtype=torch.long)
        x = self.row_embed(r) + self.col_embed(c)
        y = self.net(x)
        return y
    
    
    
class ActorCritic(BaseAgent):
    
    def __init__(self, env:Env):
        super().__init__(env)
        self.gamma = 0.9
        self.policynet = PolicyNet(env)
        self.policynet.to(device=device)
        self.qnet = QNet(env)
        self.qnet.to(device=device)
        self.n_episode = 1000
        self.optimizer = torch.optim.Adam(self.policynet.parameters(), lr=0.01)
        self.qnet_optimizer =  torch.optim.Adam(self.qnet.parameters(), lr=0.01)
        self.max_dis = 100
        self.qnet_critic = nn.MSELoss()
        
    def take_action(self, r, c):
        self.policynet.eval()
        logit = self.policynet(r, c)
        action_dist = torch.distributions.Categorical(logits=logit)
        a = action_dist.sample()
        self.policynet.train()
        return a.item()
    
    
    def train_policygradient(self):
        self.policynet.train()
        self.qnet.train()
        for i_episode in tqdm(range(self.n_episode)):
            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            done = False
            dis = 0
            a = self.take_action(r, c)
            while not done:
                nr, nc, reward, done = self.env.P[(r, c, a)]
                na = self.take_action(r, c)
                td_pred = self.qnet(r, c)[0, a]
                td_target = reward + self.gamma * self.qnet(nr, nc).detach()[0, na]
                qnet_loss = self.qnet_critic(td_pred, td_target)
                self.optimizer.zero_grad()
                self.qnet_optimizer.zero_grad()
                qnet_loss.backward()
                self.qnet_optimizer.step()
                
                actor_loss = - td_pred.detach() * F.log_softmax(self.policynet(r, c), dim=-1)[0, a]
                actor_loss.backward()
                self.optimizer.step()
                r, c, a = nr, nc, na
                dis += 1
                if dis>self.max_dis:
                    done = True
            self.env.visual_policy(self.get_policy())
    
    def get_policy(self):
        self.policynet.eval()
        index = np.asarray([[r, c] for r in range(self.env.n_row) for c in range(self.env.n_col)]).reshape(-1, 2)
        policy = self.policynet(index[:, 0], index[:, 1]).argmax(-1).reshape(self.env.n_row, self.env.n_col)
        return toNumpy(policy)
                
                
        
def main():
    env = Env()
    agent = ActorCritic(env)
    agent.train_policygradient()


main()