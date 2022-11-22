import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_model.plant_beam_model import PlantBeamModel

from . import Simulation

class IsaacGymPlantEnv(gym.Env):
    """An isaacgym plant environment for OpenAI gym"""

    def __init__(self):

        self.S = Simulation()
        
        self.action_space = spaces.Discrete(18)
        self.dtheta = 0.05

        len_von_mises = len(self.S.find_von_mises())

        # obs_low = np.concatenate((np.zeros(len_von_mises), \
        #                             self.S.franka_lower_limits, \
        #                             np.zeros(len(self.S.update_plant_pose_tensor()))))

        # obs_high = np.concatenate((np.ones(len_von_mises)*10e10,\
        #                             self.S.franka_upper_limits,\
        #                             np.ones(len(self.S.update_plant_pose_tensor()))*10))

        obs_low = np.concatenate((self.S.franka_lower_limits, \
                                    np.zeros(len(self.S.update_plant_pose_tensor()))))

        obs_high = np.concatenate((self.S.franka_upper_limits,\
                                    np.ones(len(self.S.update_plant_pose_tensor()))*10))

        # Obs: von mises stress, kinematic tensor (franka pose, sb pose)
        self.observation_space = spaces.Box(low = obs_low, high = obs_high)

        self.steps = 0

    def _take_action(self, action):
        action -= len(self.S.franka_lower_limits)
        if np.sign(action) != -1:
            action+=1

        self.S.target_angles[abs(action)-1] += self.dtheta*np.sign(action)
        self.S.command_franka_angles(self.S.target_angles)
        self.steps+=1

    def step(self, action):
        self._take_action(action)
        done = self.S.red_index>0
        reward = 0
        if done:
            reward =1
            print('Hit the jackpot')
        if self.steps > 100:
            done = True
            print('Hit step limit yeet')

        obs = self._next_observation()

        return obs, reward, done, {}

    def _next_observation(self):
        return np.array(np.concatenate((self.S.current_angles, self.S.plant_pose)))

    def reset(self):
        self.steps = 0
        self.S.command_franka_angles(np.zeros(9))
        return self._next_observation()
