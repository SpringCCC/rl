
import numpy as np



class Env():

    def __init__(self, n_row=5, n_col=5):
        self.n_row = n_row
        self.n_col = n_col

        self.n_action = 4
        self.action = {0:[-1, 0], 1:[1, 0], 2:[0, -1], 3:[0, 1]}
        self.r_step = -1
        self.r_dst = 10
        self.r_obstacle = -5
        self.init_env()
        self.create_P()
        self.reset()


    def init_env(self):
        self.obstacle_pos = [[1,1], [1,2], [2,2], [3,1], [3,3], [4,1]]
        self.dst_pos = [[3,2]]
        

    def step(self, agent_row, agent_col, a):
        nr, nc, reward, done = self.P[(agent_row, agent_col, a)]
        self.agent_pos = [nr, nc]
        return nr, nc, reward, done

    def create_P(self):
        self.P = {}
        for r in range(self.n_row):
            for c in range(self.n_col):
                for a in range(self.n_action):
                    done = False
                    nr, nc = r + self.action[a][0], c + self.action[a][1]
                    if not (0<=nr<self.n_row and 0<=nc<self.n_col):
                        nr, nc = r, c
                        reward = self.r_step
                    elif [nr, nc] in self.obstacle_pos:
                        nr, nc = r, c
                        reward = self.r_obstacle
                    elif [nr, nc] in self.dst_pos:
                        reward = self.r_dst
                        done = True
                    else:
                        reward = self.r_step
                    self.P[(r,c,a)] = (nr, nc, reward, done)


    def reset(self):
        while True:
            r = np.random.randint(self.n_row)
            c = np.random.randint(self.n_row)
            self.agent_pos = [r, c]
            if self.check_terminal(self.agent_pos):
                continue
            else:
                break

    def check_terminal(self, pos):
        if pos in self.dst_pos:
            return True
        return False
    

    def visual_policy(self, best_policy):

        arrow_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arr_arrow = np.vectorize(arrow_dict.get)(best_policy)
        dst_r, dst_c = self.dst_pos[0]
        arr_arrow[dst_r, dst_c] = r"★"   # G 表示Goal（终点）
        for zr, zc in self.obstacle_pos:
            arr_arrow[zr, zc] = "■"     # X 表示障碍
        print(f"best_policy:\n")
        print(arr_arrow)


class BaseAgent():

    def __init__(self, env:Env):
        self.env = env
        self.gamma = 0.9