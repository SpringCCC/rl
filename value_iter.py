import numpy as np
from base import Env,BaseAgent

class Agent_value_iter(BaseAgent):

    def __init__(self, env:Env):
        super().__init__(env)
        self.n_row, self.n_col = env.n_row, env.n_col
        self.v_table = np.zeros((env.n_row, env.n_col))
        self.delta_thresh = 0.01

    def take_action(self):
        pass
    

    def train_value_iter(self):

        converge  = False
        while not converge:
            delta = 0
            for r in range(self.env.n_row):
                for c in range(self.env.n_col):
                    new_v = []
                    old_v = self.v_table[r, c]
                    for a in range(self.env.n_action):
                        nr, nc, reward, done = self.env.step(r, c, a)
                        v = reward + self.gamma * self.v_table[nr, nc]
                        new_v.append(v)
                    self.v_table[r, c] = max(new_v)
                    delta = max(abs(self.v_table[r, c] - old_v), delta)
            if delta < self.delta_thresh:
                converge = True
            self.env.visual_policy(self.get_best_policy())

    

    def get_best_policy(self):

        best_policy = np.zeros((self.env.n_row, self.env.n_col))
        for r in range(self.env.n_row):
                for c in range(self.env.n_col):
                    new_v = []
                    for a in range(self.env.n_action):
                        nr, nc, reward, done = self.env.P[(r, c, a)]
                        new_v.append(reward + self.gamma * self.v_table[nr, nc])
                    best_policy[r, c] = np.asarray(new_v).argmax()

        return best_policy


def main():
    env = Env()
    agent = Agent_value_iter(env)
    agent.train_value_iter()

main()