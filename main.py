from markov_env import markov
import matplotlib.pyplot as plt
from dqn_transfer import TransferDQN as DQN
from dqn_transfer import Net
import settings as st
import torch

env = markov()

# Variables to store the rewards and losses
average_reward = []
episode_losses = []

for i in range(20):
    dqn = DQN()
    s = env.reset()  # 重置环境
    slot = 0
    flag = 0

    while True:
        a = dqn.choose_action(s)  # 选择动作
        s_, r = env.step(a)  # 与环境交互并获取下一个状态和奖励

        # 更新平均奖励
        for j in range(a % st.MAX_SLOT + 1):
            if i == 0:
                if slot < st.END_SLOT:
                    average_reward.append(r)
                    slot += 1
            else:
                if slot < st.END_SLOT:
                    average_reward[slot] = (average_reward[slot] * i + r) / (i + 1)
                    slot += 1

        print(f"episode:{i}, slot:{slot}, reward:{r}, channel:{a // st.MAX_SLOT}, time:{a % st.MAX_SLOT}")

        # 存储经验
        dqn.store_transition(s, a, r, s_, slot)

        # 更新状态
        s = s_

        # 训练神经网络
        if dqn.memory_counter > st.BATCH_SIZE:
            if slot >= st.CHANGE_SLOT:
                # 初始化经验池采样概率
                dqn.initial_probabilities(slot)
            for it in range(st.i_max):
                dqn.learn(slot)

            dqn.target_net = dqn.eval_net

        # 切换环境
        if slot >= st.CHANGE_SLOT and flag == 0:
            flag = 1
            env.change()

            # 重置 DQN 的网络结构
            dqn.eval_net = Net().to(dqn.device)
            dqn.target_net = Net().to(dqn.device)
            dqn.optimizer = torch.optim.Adam(dqn.eval_net.parameters(), lr=st.LR)
            dqn.e = 0.95

        if slot >= st.END_SLOT:
            break

# 训练完成后，绘制奖励图形
plt.figure(figsize=(80, 40))

# 绘制奖励
plt.subplot(1, 1, 1)
plt.plot(average_reward)
plt.title('Average Reward')
plt.xlabel('Slot')
plt.ylabel('Average Reward')

plt.tight_layout()
plt.show()

