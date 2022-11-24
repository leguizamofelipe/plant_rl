import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from basic_model.plant_beam_model import PlantBeamModel

from . import Simulation
from ICM import ICM
from ICM.memory import Memory

class IsaacGymPlantEnv(gym.Env):
    """An isaacgym plant environment for OpenAI gym"""

    def __init__(self, headless = False):
        
        self.action_space = spaces.Discrete(18)
        self.dtheta = 0.05

        self.S = Simulation()

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

        self.memory = Memory()

        # ICM startup
        import torch
        self.ICM = ICM(self.observation_space.shape[0], self.action_space.n)

        self.icm_optimizer = torch.optim.Adam(self.ICM.parameters(), lr=1e-4)

        self.icm_steps = 20
        self.ep_steps = 0
        self.total_steps = 0

    def _take_action(self, action):
        action -= len(self.S.franka_lower_limits)
        if np.sign(action) != -1:
            action+=1

        self.S.target_angles[abs(action)-1] += self.dtheta*np.sign(action)
        self.S.command_franka_angles(self.S.target_angles)
        self.ep_steps+=1
        self.total_steps+=1
        # print(self.total_steps)

    def step(self, action):
        obs_0 = self._next_observation()
        self._take_action(action)
        obs_1 = self._next_observation()

        done = self.S.red_index>0
        reward = 0

        if done:
            reward =1
            print('Hit the jackpot')
        if self.ep_steps >= 256:
            done = True
            print('Hit step limit yeet')
        self.memory.remember(obs_0, action, reward, obs_1, self.value[0][0].item(), self.logprobs[0].item())

        # Update ICM
        if self.ep_steps % self.icm_steps == 0 or done:
            states, actions, rewards, new_states, values, log_probs = self.memory.sample_memory()
            intrinsic_reward, Li, Lf = self.ICM.calc_loss(states, new_states, actions)

            self.icm_optimizer.zero_grad()
            (Li+Lf).backward()

            self.icm_optimizer.step()
            self.reward_buffer = intrinsic_reward
            self.memory.clear_memory()

        return obs_1, reward, done, {}

    def _next_observation(self):
        return np.array(np.concatenate((self.S.current_angles, self.S.plant_pose)))

    def reset(self):
        self.ep_steps = 0
        self.S.command_franka_angles(np.zeros(9))
        return self._next_observation()
