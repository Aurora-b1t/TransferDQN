BATCH_SIZE = 64 # 样本数量
GAMMA = 0.9 # 奖励折扣
i_max = 10 # 迭代次数
Lambda = 10.0 # 比例系数
LR = 0.001 # 学习率
Beta = 50 # 比例系数
EPSILON_MAX = 0.99 # 最大 epsilon

MAX_SLOT = 30 # 最大时隙数
N_state = 8 # 状态空间
N_action = 8*MAX_SLOT # 动作空间

CHANGE_SLOT = 12000 # 改变时隙
END_SLOT = 20000 # 结束时隙