
import numpy as np


class Env():

    def __init__(self, gridSize, dst_pos, zhangai_pos):
        self.step_reward = -1
        self.dst_reward = 10
        self.zhangai_reward = -2
        self.gridSize = gridSize
        self.m = gridSize[0] # row
        self.n = gridSize[1] # cool
        self.grid = np.zeros(gridSize)
        self.n_actions = 4
        self.step_reward = -1
        self.mv = {
            0: [-1, 0],#上 x,y
            1: [1, 0],#下
            2: [0, -1],#左
            3: [0, 1],#右
        }
        self.dst_pos = dst_pos
        self.zhangai_pos = zhangai_pos
        self.P = self.init_P()
        self.v = np.zeros(gridSize)
        self.gamma = 0.9
        self.delta_thresh = 0.0001
        self.eps = 0.5
        self.n_eplisode = 1000
        self.alpha = 0.1
        self.visit = np.zeros(gridSize, dtype=np.int32)
    

    def check_terminal(self, pos):
        if pos in self.dst_pos:
            return True
        else:
            return False
        

    def init_P(self):
        P = {}
        for i in range(self.m): 
            for j in range(self.n):
                for k in range(self.n_actions):
                    next_i = i + self.mv[k][0]
                    next_j = j + self.mv[k][1]
                    if not (0 <= next_i < self.m and 0 <= next_j < self.n):
                        P[(i, j, k)] = (i, j, self.step_reward)
                    else:
                        if [next_i, next_j] in self.dst_pos:
                            P[(i, j, k)] = (next_i, next_j, self.dst_reward)
                        elif [next_i, next_j] in self.zhangai_pos:
                            P[(i, j, k)] = (i, j, self.zhangai_reward)
                        else:
                            P[(i, j, k)] = (next_i, next_j, self.step_reward)
        return P

    def sarsa_iter(self):
        for t in range(self.n_eplisode):
            print(f"这是第{t}条轨迹")
            self.eps -= t//10
            self.eps = max(0.1, self.eps)
            i, j  = 0, 0
            while True:
                i, j = np.random.randint(0, self.m-1),  np.random.randint(0, self.n-1)
                if self.check_terminal([i, j]):
                    continue
                else:
                    break

            done  = False
            while not done:
                self.visit[i, j] += 1
                if np.random.uniform() < self.eps:#小概率
                    a = np.random.randint(0, self.n_actions-1)
                else:
                    new_v = []
                    for k in range(self.n_actions):
                        next_i, next_j, r = self.P[(i, j, k)]
                        new_v.append(r + self.gamma*self.v[next_i, next_j])
                    a = np.asarray(new_v).argmax()
                next_i, next_j, r = self.P[(i, j, a)]
                self.v[i, j] += self.alpha * (r + self.gamma*self.v[next_i, next_j] - self.v[i, j])
                i, j = next_i, next_j
                if self.check_terminal([i, j]):
                    done = True

        
        best_policy = np.zeros(self.gridSize)
        for i in range(self.m):
            for j in range(self.n):
                new_v = []
                for k in range(self.n_actions):
                    next_i, next_j, r = self.P[(i, j, k)]
                    new_v.append(r+self.gamma*self.v[next_i, next_j])
                best_policy[i, j] = np.asarray(new_v).argmax()
        arrow_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arr_arrow = np.vectorize(arrow_dict.get)(best_policy)
        print(f"best_policy:\n")
        print(arr_arrow)
        print(self.v)
        print("self.visit")
        print(self.visit)





            
def main():
    dst = [[3, 2]]
    zhangai = [[1,1], [1, 2], [2,2], [3,1], [3, 3], [4, 1]]
    gridSize = [5, 5]
    env = Env(gridSize, dst, zhangai)
    env.sarsa_iter()
    
                    
main()