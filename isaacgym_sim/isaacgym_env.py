import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class IsaacGymPlantEnv(gym.Env):
    """An isaacgym plant environment for OpenAI gym"""

    def __init__(self, simulation, env_n = 0, headless = False):
        
        self.action_space = spaces.Discrete(18)
        self.dtheta = 0.05

        self.S = simulation
        self.env_n = env_n
        self.headless = headless

        # len_von_mises = len(self.S.find_von_mises())

        # obs_low = np.concatenate((self.S.franka_lower_limits, \
        #                             np.zeros(len(self.S.update_plant_pose_tensor()))))

        # obs_high = np.concatenate((self.S.franka_upper_limits,\
        #                             np.ones(len(self.S.update_plant_pose_tensor()))*10))
        obs_low = self.S.franka_lower_limits
        obs_high = self.S.franka_upper_limits

        # Obs: von mises stress, kinematic tensor (franka pose, sb pose)
        self.observation_space = spaces.Box(low = obs_low, high = obs_high)

        self.icm_steps = 20
        self.ep_steps = 0
        self.total_steps = 0

    def _take_action(self, action):
        action -= len(self.S.franka_lower_limits)
        if np.sign(action) != -1:
            action+=1

        self.S.target_angles[self.env_n][abs(action)-1] += self.dtheta*np.sign(action)

        self.S.set_franka_angles(self.S.target_angles[self.env_n], self.env_n)
        
        self.ep_steps+=1
        self.total_steps+=1

    def step(self, action):
        self._take_action(action)
        obs_1 = self._next_observation()

        # done = self.S.red_indexes[env_n]>0
        done = False
        reward = 0

        if done:
            reward =1
            print('Hit the jackpot')
        if self.ep_steps >= 256:
            done = True
            print('Hit step limit')

        return obs_1, reward, done, {}

    def _next_observation(self):
        # return np.array(np.concatenate((self.S.current_angles[self.env_n], self.S.plant_pose[self.env_n])))
        return self.S.get_franka_angles(self.env_n)

    def reset(self):
        self.ep_steps = 0
        self.S.set_franka_angles(np.zeros(9), self.env_n)
        return self._next_observation()
