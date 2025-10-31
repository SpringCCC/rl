import numpy as np
from base import Env, BaseAgent


class Qlearn(BaseAgent):
    
    def __init__(self, env):
        super().__init__(env)
        self.eps = 0.1
        self.q_table = np.zeros((self.env.n_row, self.env.n_col, self.env.n_action), dtype=np.float32)
        self.n_episode = 1000
        self.max_dis = 100
        self.alpha = 0.1
        
        
        
    def take_action(self, r, c):
        if np.random.uniform()<self.eps:
            a = np.random.randint(self.env.n_action)
        else:
            a = self.q_table[r, c].argmax()
        return a
    
    def train_qlearn(self):
        for i_episode in range(self.n_episode):
            dis = 0
            done = False
            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            while not done:
                a = self.take_action(r, c)
                nr, nc, reward, done = self.env.P[(r, c, a)]
                td_error = reward + self.gamma*max(self.q_table[nr, nc])*(1-done) - self.q_table[r, c, a]
                self.q_table[r,c,a] += self.alpha*td_error
                r, c = nr, nc
                dis += 1
                if dis>self.max_dis:
                    done = True
            self.env.visual_policy(self.get_policy())
    
    def get_policy(self):
        return self.q_table.argmax(-1)
    
    
def main():
    env = Env()
    agent = Qlearn(env)
    agent.train_qlearn()
    agent.env.visual_policy(agent.get_policy())
            
main()