import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import settings as st
# GPU设置
if torch.cuda.is_available():
    device = "cuda"
    print(torch.cuda.get_device_name(0))
else:
    device = "cpu"

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(st.N_state, 20)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(20, 20)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(20, 20)
        self.fc3.weight.data.normal_(0, 0.1)
        # 输出层：两个分支分别输出动作空间的分布
        self.action_head = nn.Linear(20, st.N_action)
        self.action_head.weight.data.normal_(0, 0.1)


    def forward(self, x):
        x = F.relu(self.fc1(x))  # 第一层 ReLU 激活
        x = F.relu(self.fc2(x))  # 第二层 ReLU 激活
        x = F.relu(self.fc3(x))  # 第三层 ReLU 激活

        action = self.action_head(x)  # 输出动作空间
        return action

class TransferDQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = []  # 经验池
        self.source_memory = []  # 源域经验
        self.target_memory = []  # 目标域经验
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=st.LR)
        self.loss_func = nn.MSELoss().to(device)
        self.device = device
        self.e = 0.95

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)
        if np.random.uniform() < self.e and self.memory_counter > 100:
            action_value = self.eval_net(x)
            action = torch.argmax(action_value, dim=1).item()
        else:
            action = np.random.randint(0, st.N_action)
        return action

    def store_transition(self, s, a, r, s_, slot):
        transition = np.hstack((s, a, r, s_))
        source_flag = 0 if slot <= st.CHANGE_SLOT else 1
        transition = np.append(transition, source_flag)

        # 初始化采样概率
        sampling_prob = 1.0 if slot <= st.CHANGE_SLOT else 0.0
        self.memory.append({'transition': transition, 'prob': sampling_prob})
        self.memory_counter += 1

    def update_probabilities(self, slot):
        # 根据源域和目标域经验池大小初始化概率
        self.source_memory = [exp for exp in self.memory if exp['transition'][-1] == 0]
        self.target_memory = [exp for exp in self.memory if exp['transition'][-1] == 1]
        source_size = len(self.source_memory)
        target_size = len(self.target_memory)

        # 计算初始概率
        for exp in self.source_memory:
            exp['prob'] = 1.0 / (source_size + st.Lambda * target_size)
        for exp in self.target_memory:
            exp['prob'] = st.Lambda / (source_size + st.Lambda * target_size)

    def sample_experiences(self):
        probabilities = np.array([exp['prob'] for exp in self.memory], dtype=np.float64)
        probabilities /= probabilities.sum()  # 归一化

        sampled_indices = np.random.choice(
            len(self.memory), size=st.BATCH_SIZE, replace=False, p=probabilities
        )
        # 忽略 source_flag
        batch = [self.memory[idx]['transition'][:-1] for idx in sampled_indices]
        batch = np.array(batch)
        return torch.FloatTensor(np.stack(batch)).to(self.device)

    def calculate_td_error(self, transition):
        transition = transition[:-1]  # 忽略最后的 source_flag

        # 提取状态、动作、奖励、下一状态
        b_s = torch.FloatTensor(transition[:st.N_state]).to(self.device).unsqueeze(0)  # 状态
        b_a = torch.LongTensor([transition[st.N_state]]).to(self.device).view(-1, 1)  # 动作
        b_r = torch.FloatTensor([transition[st.N_state + 1]]).to(self.device)  # 奖励
        b_s_ = torch.FloatTensor(transition[-st.N_state:]).to(self.device).unsqueeze(0)  # 下一状态

        # 计算 Q 值和 TD-Error
        q_eval = self.eval_net(b_s).gather(1, b_a)  # 评估网络的 Q 值
        q_next = self.target_net(b_s_).max(1)[0].view(-1, 1)  # 目标网络的 Q 值
        q_target = b_r + st.GAMMA * q_next  # 目标 Q 值
        td_error = torch.abs(q_eval - q_target).item()  # 计算 TD-Error

        return td_error

    def update_td_errors(self):
        # 遍历经验池，计算每条经验的 TD-error 并更新采样概率
        for exp in self.memory:
            td_error = self.calculate_td_error(exp['transition'])
            if exp['transition'][-1] == 0:  # Source memory
                exp['prob'] *= (1 + td_error / st.Beta)
            else:  # Target memory
                exp['prob'] /= (1 + td_error / st.Beta)

        # 归一化采样概率，确保总和为 1
        total_prob = sum(exp['prob'] for exp in self.memory)
        for exp in self.memory:
            exp['prob'] /= total_prob

    def learn(self, slot):
        self.learn_step_counter += 1

        batch = self.sample_experiences()

        b_s = batch[:, :st.N_state]
        b_a = batch[:, st.N_state:st.N_state + 1].long()
        b_r = batch[:, st.N_state + 1:st.N_state + 2]
        b_s_ = batch[:, -st.N_state:]

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).max(1)[0].view(-1, 1)
        q_target = b_r + st.GAMMA * q_next

        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新 TD-Error 和采样概率
        if slot >= st.CHANGE_SLOT:
            self.update_td_errors()
