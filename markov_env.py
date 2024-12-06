import gym
from gym import spaces
import numpy as np
from gym.spaces import Discrete
import settings as st


class markov(gym.Env):

    def __init__(self):
        self.idle_prob = np.array([0.85, 0.8, 0.9, 0.5, 0.7, 0.95, 0.7, 0.5])
        self.busy_prob = np.array([0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5])
        self.observation_space = spaces.Box(low=0, high=1, shape=(8,), dtype=np.int32)
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        self.action_space = Discrete(8*st.MAX_SLOT)
        self.sensing_slots = 1


    def step(self, action):
        t_reward = 0
        if action%st.MAX_SLOT != 0:
            for i in range(action%(st.MAX_SLOT)):
                if self.state[action//st.MAX_SLOT] == 0:
                    t_reward += 1
                elif self.state[action//st.MAX_SLOT] == 1:
                    t_reward -= 3
                for j in range(8):
                    if self.state[j] == 0:
                        if np.random.rand() > self.idle_prob[j]:
                            self.state[j] = 1
                    else:
                        if np.random.rand() > self.busy_prob[j]:
                            self.state[j] = 0

        t_reward -= 2*self.sensing_slots

        return self.state, t_reward

    def reset(self):
        self.state = np.random.choice([0, 1], size=8, p=[0.5, 0.5])
        self.idle_prob = np.array([0.85, 0.8, 0.9, 0.5, 0.7, 0.95, 0.7, 0.5])
        self.busy_prob = np.array([0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5])
        return self.state

    def change(self):
        self.idle_prob = np.array([0.75, 0.8, 0.9, 0.5, 0.7, 0.7, 0.7, 0.5])
        self.busy_prob = np.array([0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5])
        #self.idle_prob = np.array([0.55, 0.5, 0.5, 0.55, 0.5, 0.6, 0.3, 0.5])
        #self.busy_prob = np.array([0.85, 0.8, 0.9, 0.5, 0.7, 0.95, 0.7, 0.5])
