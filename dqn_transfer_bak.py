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


# 定义DQN类
class TransferDQN(object):
    def __init__(self):
        # 创建评估网络和目标网络
        self.eval_net, self.target_net = Net().to(device), Net().to(device)
        self.learn_step_counter = 0  # 学习步数记录
        self.memory_counter = 0      # 记忆量计数
        self.source_memory = []  # 使用 list 存储经验
        self.target_memory = []
        self.optimzer = torch.optim.Adam(self.eval_net.parameters(), lr=st.LR)
        self.loss_func = nn.MSELoss().to(device)  # 使用均方损失函数
        self.device = device
        self.e = 0.95
        self.p_i = 0.5
        self.p_j = 0.5

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0).to(device)  # 将状态转换为 tensor 并添加批次维度

        # epsilon-greedy 策略
        if self.e < st.EPSILON_MAX and self.memory_counter % 20 == 0:
            self.e = 0.998*self.e+0.002  # epsilon 随时间增加
            print(f"epsilon:{self.e}")

        if np.random.uniform() < self.e and self.memory_counter > 100:
            # 使用评估网络预测当前状态的 Q 值
            action_value = self.eval_net(x)
            action = torch.argmax(action_value, dim=1).item()  # 获取最大 Q 值对应的动作
        else:
            # 随机选择动作
            action = np.random.randint(0, st.N_action)

        return action

    def store_transition(self, s, a, r, s_, slot):
        # 合并状态、动作、奖励和下一状态，并记录该样本的来源
        transition = np.hstack((s, a, r, s_))
        # 添加来源标识，0 代表源域，1 代表目标域
        source_flag = 0 if slot <= st.CHANGE_SLOT else 1
        transition = np.append(transition, source_flag)  # 将来源标识添加到转移样本中

        if slot <= st.CHANGE_SLOT:
            self.source_memory.append(torch.tensor(transition, device=device))
        else:
            self.target_memory.append(torch.tensor(transition, device=device))

        self.memory_counter += 1

    def learn(self, slot):
        self.learn_step_counter += 1

        # 随机采样一个批次的记忆
        if slot < st.CHANGE_SLOT:
            batch = np.random.choice(len(self.source_memory), st.BATCH_SIZE, replace=False)
            b_memory = torch.stack([self.source_memory[idx] for idx in batch])
        else:
            batch_source = []  # 存储源域采样的结果
            batch_target = []  # 存储目标域采样的结果

            # 按照概率 p_i 选择从源域还是目标域采样
            for k in range(st.BATCH_SIZE):
                if np.random.random() < self.p_i or len(self.target_memory) == 0:
                    # 从源域采样
                    batch_source.append(np.random.choice(len(self.source_memory), 1, replace=False)[0])
                else:
                    # 从目标域采样
                    batch_target.append(np.random.choice(len(self.target_memory), 1, replace=False)[0])

            # 现在从 batch_source 和 batch_target 选择对应的样本
            b_memory_source = torch.tensor([]).to(device)
            if len(batch_source) != 0:
                b_memory_source = torch.stack([self.source_memory[idx] for idx in batch_source])
            b_memory_target = torch.tensor([]).to(device)
            if len(batch_target) != 0:
                b_memory_target = torch.stack([self.target_memory[idx] for idx in batch_target])

            # 合并源域和目标域的样本，得到一个完整的批次
            b_memory = torch.cat([b_memory_source, b_memory_target], dim=0)

        # 获取批次中的状态、动作、奖励、下一个状态和来源标识
        b_s = b_memory[:, :st.N_state]  # 当前状态
        b_a = b_memory[:, st.N_state:st.N_state + 1].long()  # 当前动作
        b_r = b_memory[:, st.N_state + 1:st.N_state + 2]  # 当前奖励
        b_s_ = b_memory[:, -st.N_state:]  # 下一个状态
        b_source_flag = b_memory[:, -1].long()  # 记录来源标识（源域或目标域）

        # 转换成 PyTorch 张量并确保它们在相同的设备上
        b_s = b_s.clone().detach().to(device).float()  # 当前状态
        b_a = b_a.clone().detach().to(device).long()  # 当前动作
        b_r = b_r.clone().detach().to(device).float()  # 当前奖励
        b_s_ = b_s_.clone().detach().to(device).float()  # 下一个状态

        # 计算 q_eval（评估网络的输出）
        q_eval = self.eval_net(b_s).gather(1, b_a)

        # 计算 q_target（目标网络的输出）
        q_next = self.target_net(b_s_)
        q_target = b_r + st.GAMMA * q_next.max(1)[0].view(-1, 1)

        # 计算损失（均方误差损失）
        loss = self.loss_func(q_eval, q_target)
        self.optimzer.zero_grad()
        loss.backward()
        self.optimzer.step()

        if slot>st.CHANGE_SLOT:
            td_error = (torch.abs(q_eval - q_target).detach()).to('cuda').to(torch.float32)
            # 记录每个样本的 TD error 以及其来源
            for i in range(len(td_error)):
                if b_source_flag[i].item() == 0:
                    self.p_i=self.p_i*(1-td_error[i].item()/st.Beta)
                else:
                    self.p_j=self.p_j/(1-td_error[i].item()/st.Beta)
            self.p_i=self.p_i/(self.p_i+self.p_j)
            self.p_j=1-self.p_i


        return loss.item()  # 返回损失值，用于后续的绘图
