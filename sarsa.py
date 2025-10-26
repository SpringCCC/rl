
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
        self.gamma = 1.0
        self.delta_thresh = 0.0001
        self.eps = 0.1
        self.base_prob = self.eps / self.n_actions
        self.max_prob = 1 - (self.n_actions -1)*self.base_prob
        self.policy_prob = self.make_prob()
    
    def make_prob(self):
        probs = {}
        for i in range(self.n_actions):
            p = [self.base_prob]* self.n_actions
            p[i] =  self.max_prob
            probs[i] = np.asarray(p)
        return probs

    def check_terminal(self, pos):
        if pos in self.dst_pos + self.zhangai_pos:
            return True
        else:
            return False
        
    def gen_exp()
    

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

    def value_iter(self):
        cnt = 0
        converge = False
        while not converge:
            cnt += 1
            delta = 0
            for j in range(self.m):
                for i in range(self.n):
                    if self.check_terminal([i, j]):
                        self.v[i, j] = 0
                        continue
                    old_v = self.v[i, j]
                    new_v = []
                    for k in range(self.n_actions):
                        next_i, next_j, r = self.P[(i, j, k)]
                        if next_i==3 and next_j==2:
                            sadad=2
                        new_v.append(r+self.gamma*self.v[next_i, next_j])
                    self.v[i, j] = max(new_v)
                    delta = max(delta, abs(self.v[i, j] - old_v))
                    converge = True if delta < self.delta_thresh else False
            print(f"epoch:{cnt}\tdelta={delta}")
        # 迭代完成
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





            
def main():
    dst = [[3, 2]]
    zhangai = [[1,1], [1, 2], [2,2], [3,1], [3, 3], [4, 1]]
    gridSize = [5, 5]
    env = Env(gridSize, dst, zhangai)
    env.value_iter()
    
                    
main()