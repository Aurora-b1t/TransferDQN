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

#保存图片
def save_plot(data, filename, title, xlabel, ylabel):
    """
    保存绘图为图片文件
    :param data: 绘制的数据
    :param filename: 保存的文件名
    :param title: 图表标题
    :param xlabel: x 轴标签
    :param ylabel: y 轴标签
    """
    plt.figure(figsize=(80, 40))
    plt.plot(data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()

    # 保存图片到当前目录
    plt.savefig(filename)
    plt.close()  # 关闭当前图表，释放内存
    print(f"Plot saved to {filename}")


for i in range(1):
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
            #dqn.e = 0.95

        if slot >= st.END_SLOT:
            break

# 训练完成后，绘制奖励图形
save_plot(
    data=average_reward,
    filename="average_reward.png",  # 保存到当前目录
    title="Average Reward",
    xlabel="Slot",
    ylabel="Reward"
)


