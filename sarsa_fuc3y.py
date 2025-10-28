import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------------
# 环境类保持原样
# -----------------------------
class Env():
    def __init__(self, gridSize, dst_pos, zhangai_pos):
        self.step_reward = -1
        self.dst_reward = 10
        self.zhangai_reward = -2
        self.gridSize = gridSize
        self.m = gridSize[0]  # row
        self.n = gridSize[1]  # col
        self.n_actions = 4
        self.mv = {0: [-1,0], 1:[1,0], 2:[0,-1], 3:[0,1]}
        self.dst_pos = dst_pos
        self.zhangai_pos = zhangai_pos
        self.P = self.init_P()
        self.gamma = 0.9
        self.n_episode = 1000
        self.eps = 0.1
        self.alpha = 0.01  # 学习率用于 optimizer

    def check_terminal(self, pos):
        return pos in self.dst_pos

    def init_P(self):
        P = {}
        for i in range(self.m):
            for j in range(self.n):
                for k in range(self.n_actions):
                    next_i = i + self.mv[k][0]
                    next_j = j + self.mv[k][1]
                    if not (0 <= next_i < self.m and 0 <= next_j < self.n):
                        P[(i,j,k)] = (i,j,self.step_reward)
                    else:
                        if [next_i,next_j] in self.dst_pos:
                            P[(i,j,k)] = (next_i,next_j,self.dst_reward)
                        elif [next_i,next_j] in self.zhangai_pos:
                            P[(i,j,k)] = (i,j,self.zhangai_reward)
                        else:
                            P[(i,j,k)] = (next_i,next_j,self.step_reward)
        return P

# -----------------------------
# Q 网络 + Embedding
# -----------------------------
class QNetwork(nn.Module):
    def __init__(self, m, n, n_actions, embed_dim=16):
        super().__init__()
        self.embedding_i = nn.Embedding(m, embed_dim)
        self.embedding_j = nn.Embedding(n, embed_dim)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim*2, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, s):
        # s = [i,j]
        i = torch.tensor([s[0]], dtype=torch.long).cuda()
        j = torch.tensor([s[1]], dtype=torch.long).cuda()
        ei = self.embedding_i(i)
        ej = self.embedding_j(j)
        x = torch.cat([ei, ej], dim=-1)
        q = self.fc(x)
        return q  # [1, n_actions]

# -----------------------------
# 使用函数近似的 SARSA + 双网络
# -----------------------------
class SARSA_FA_Env(Env):
    def __init__(self, *args, target_update_freq=50, **kwargs):
        super().__init__(*args, **kwargs)
        self.q_net = QNetwork(self.m, self.n, self.n_actions).cuda()
        self.target_net = QNetwork(self.m, self.n, self.n_actions).cuda()
        self.target_net.load_state_dict(self.q_net.state_dict())  # 初始化
        self.target_net.eval()  # 目标网络只用于生成 target
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.alpha)
        self.loss_fn = nn.MSELoss()
        self.target_update_freq = target_update_freq  # 每多少步更新一次目标网络
        self.step_counter = 0

    def gen_action(self, s):
        if np.random.rand() < self.eps:
            return np.random.randint(0, self.n_actions)
        else:
            q_values = self.q_net(s)
            return torch.argmax(q_values).item()

    def sarsa_iter(self):
        for ep in range(self.n_episode):
            # 随机初始状态
            while True:
                i, j = np.random.randint(0, self.m), np.random.randint(0, self.n)
                if not self.check_terminal([i,j]):
                    break
            s = [i,j]
            a = self.gen_action(s)
            done = False
            print(f"第{ep}条轨迹")
            while not done:
                next_i, next_j, r = self.P[(s[0], s[1], a)]
                s_next = [next_i, next_j]
                done = self.check_terminal(s_next)

                if done:
                    target = torch.tensor([r], dtype=torch.float32).cuda()
                else:
                    next_a = self.gen_action(s_next)
                    with torch.no_grad():  # 使用目标网络计算 target
                        q_next = torch.max(self.target_net(s_next))
                    target = torch.tensor([r + self.gamma * q_next.item()], dtype=torch.float32).cuda()

                # Q(s,a)
                q_sa = self.q_net(s)[0, a]
                loss = self.loss_fn(q_sa, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # 每隔 target_update_freq 步更新目标网络
                self.step_counter += 1
                if self.step_counter % self.target_update_freq == 0:
                    self.target_net.load_state_dict(self.q_net.state_dict())

                s = s_next
                if not done:
                    a = next_a

            if (ep+1) % 100 == 0:
                print(f"Episode {ep+1} finished")

    def visualize_policy(self):
        # 输出每个状态的最优动作
        grid_display = np.full((self.m, self.n), '', dtype=object)
        arrow_dict = {0:"↑", 1:"↓", 2:"←",3:"→"}
        for i in range(self.m):
            for j in range(self.n):
                s = [i,j]
                if self.check_terminal(s):
                    grid_display[i,j] = '★'
                elif s in self.zhangai_pos:
                    grid_display[i,j] = '■'
                else:
                    q_values = self.q_net(s)
                    grid_display[i,j] = arrow_dict[torch.argmax(q_values).item()]
        print("Policy Grid (★目标, ■障碍):")
        for row in grid_display:
            print(' '.join(row))

# -----------------------------
# 主程序
# -----------------------------
def main():
    dst = [[3,2]]
    zhangai = [[1,1],[1,2],[2,2],[3,1],[3,3],[4,1]]
    env = SARSA_FA_Env([5,5], dst, zhangai)
    env.sarsa_iter()
    env.visualize_policy()

if __name__ == "__main__":
    main()
