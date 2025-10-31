import numpy as np
from base import Env, BaseAgent

class Agent_policy_iter(BaseAgent):

    def __init__(self, env:Env):

        super().__init__(env)
        self.pai = np.zeros((self.env.n_row, self.env.n_col), dtype=np.int32)
        self.v = np.zeros((self.env.n_row, self.env.n_col), dtype=np.float32)
        self.delta_thresh = 0.01


    def train_policy_iter(self):
        self.env.visual_policy(self.pai)
        while True:
            converge = False
            old_pai = self.pai.copy()
            while not converge:
                delta = 0
                for r in range(self.env.n_row):
                    for c in range(self.env.n_col):
                        a = self.pai[r, c]
                        old_v = self.v[r, c]
                        nr, nc, reward, done = self.env.P[(r, c, a)]
                        self.v[r, c] = reward + self.gamma * self.v[nr, nc]
                        delta = max(delta, abs(self.v[r, c]-old_v))
                if delta < self.delta_thresh:
                    converge = True
            print(self.v)
            
            # policy update
            for r in range(self.env.n_row):
                for c in range(self.env.n_col):
                    new_v = []
                    for a in range(self.env.n_action):
                        nr, nc, reward, done = self.env.P[(r, c, a)]
                        new_v.append(reward + self.gamma * self.v[nr, nc])
                    self.pai[r, c] = np.asarray(new_v).argmax()
            self.env.visual_policy(self.pai)
            if (old_pai!=self.pai).sum()==0:
                break


def main():
    env = Env()
    agent = Agent_policy_iter(env)
    agent.train_policy_iter()

main()

    


