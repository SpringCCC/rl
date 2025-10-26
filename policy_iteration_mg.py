
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
        self.policy = np.zeros(gridSize, dtype=np.int32)

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

    def policy_iter(self):
        cnt = 0
        while True:
            cnt += 1
            v_cnt = 0
            while True:
                delta = 0
                for i in range(self.m):
                    for j in range(self.n):
                        if self.check_terminal([i, j]):
                            continue
                        old_v = self.v[i, j]
                        k = self.policy[i, j]
                        next_i, next_j, r = self.P[(i, j, k)]
                        self.v[i, j] = r + self.gamma * self.v[next_i, next_j]
                        delta = max(delta, np.abs(self.v[i, j] - old_v))
                if delta < self.delta_thresh:
                    v_cnt += 1
                    print(f"v_cnt_converge = {v_cnt}")
                    break
                
            old_policy = self.policy.copy()
            for i in range(self.m):
                for j in range(self.n):
                    if self.check_terminal([i, j]):
                        continue
                    new_v = []
                    for k in range(self.n_actions):
                        next_i, next_j, r = self.P[(i, j, k)]
                        t = r + self.gamma * self.v[next_i, next_j]
                        new_v.append(t)
                    self.policy[i,j] = np.asarray(new_v).argmax()
            if (old_policy!=self.policy).sum()==0:
                break
           
        # 迭代完成
        arrow_dict = {0: "↑", 1: "↓", 2: "←", 3: "→"}
        arr_arrow = np.vectorize(arrow_dict.get)(self.policy)
        print(f"best_policy:\n")
        print(arr_arrow)
        print(self.v)

            
def main():
    dst = [[3, 2]]
    zhangai = [[1,1], [1, 2], [2,2], [3,1], [3, 3], [4, 1]]
    gridSize = [5, 5]
    env = Env(gridSize, dst, zhangai)
    env.policy_iter()
    
                    
main()