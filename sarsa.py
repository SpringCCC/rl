import numpy as np
from base import Env, BaseAgent
from tqdm import tqdm

class SARSA(BaseAgent):

    def __init__(self, env):
        super().__init__(env)
        self.eps = 0.1
        self.q_table = np.zeros((self.env.n_row, self.env.n_col, self.env.n_action), dtype=np.float32)
        self.n_episode = 100
        self.max_length = 50
        self.alpha = 0.1


    def take_action(self, r, c):
        if np.random.uniform()<self.eps:
            a = np.random.randint(self.env.n_action)
        else:
            a = self.q_table[r, c].argmax()
        return a
    
    def train_sarsa_iter(self):

        for i_episode in tqdm(range(self.n_episode)):
            self.env.reset()
            r, c = self.env.agent_pos[0], self.env.agent_pos[1]
            done = False
            dis = 0
            a = self.take_action(r, c)
            while not done:
                nr, nc, reward, done = self.env.P[(r, c, a)]
                na = self.take_action(nr, nc)
                td_error = reward + self.gamma * self.q_table[nr, nc, na]*(1-done) - self.q_table[r, c, a]
                self.q_table[r, c, a] += self.alpha * td_error
                dis += 1
                r, c = nr, nc
                a = na
                if dis > self.max_length:
                    done = True
            self.env.visual_policy(self.get_policy())
    

    def get_policy(self):
        return self.q_table.argmax(axis=-1)


def main():
    env = Env()
    agent = SARSA(env)
    agent.train_sarsa_iter()

main()