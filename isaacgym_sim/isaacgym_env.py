import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from isaacgym import gymapi
import cv2

class IsaacGymPlantEnv(gym.Env):
    """An isaacgym plant environment for OpenAI gym"""

    def __init__(self, simulation, env_n = 0, headless = False, observation_mode = 'Fully Observable', action_mode = 'All Joints'):
        # USER DEFINITIONS
        self.icm_steps = 20
        ####################       
        
        self.action_mode = action_mode
        self.observation_mode = observation_mode

        if action_mode == 'All Joints':
            self.action_space = spaces.Discrete(14)
        elif action_mode == 'One Joint':
            self.action_space = spaces.Discrete(3)

        self.dtheta = 0.07

        self.S = simulation
        self.env_n = env_n
        self.headless = headless

        self.clearances = 0
        # self.no_contacts = 0

        if observation_mode == 'Fully Observable':
            obs_low = np.concatenate([-5*np.ones(self.S.n_plant_xyz_points),self.S.franka_lower_limits])
            obs_high = np.concatenate([5*np.ones(self.S.n_plant_xyz_points),self.S.franka_upper_limits])
        elif observation_mode == 'Grayscale Image':
            obs_low = np.zeros(self.S.grayscale_shape[0]*self.S.grayscale_shape[1])
            obs_high = 255*np.ones(self.S.grayscale_shape[0]*self.S.grayscale_shape[1])

        # Obs: von mises stress, kinematic tensor (franka pose, sb pose)
        self.observation_space = spaces.Box(low = obs_low, high = obs_high)

        self.ep_steps = 0
        self.total_steps = 0

    def _take_action(self, action):
        if self.action_mode=='All Joints':
            action -= len(self.S.franka_lower_limits)
            if np.sign(action) != -1:
                action+=1
            self.S.target_angles[self.env_n][abs(action)-1] += self.dtheta*np.sign(action)

        elif self.action_mode == 'One Joint':
            action -= 1
            self.S.target_angles[self.env_n][0] += self.dtheta*action

        self.S.set_franka_angles_target(self.S.target_angles[self.env_n], self.env_n)

        self.ep_steps+=1
        self.total_steps+=1

    def step(self, action):
        # self._take_action(action)
        obs_1 = self._next_observation()

        done = self.S.red_indexes[self.env_n]>0
        reward = 0

        if min(abs(self.S.current_angles[self.env_n][0:7] - self.S.franka_upper_limits[0:7])) < self.dtheta*3 or min(abs(self.S.current_angles[self.env_n][0:7] - self.S.franka_lower_limits[0:7])) < self.dtheta*3:
            reward += -5
            print('Hit limits!')

        # if self.S.current_angles[self.env_n][0]<0.1:
        #     reward-=5
        # else:
        #     print('Made contact')

        if self.S.top_ten_mean > 250000:
            reward-=5

        if done:
            reward+=10
            self.clearances+=1
            # print('Cleared occlusion')
        if self.ep_steps>=256:
            done=True
            # print('Hit step limit')

        return obs_1, reward, done, {}

    def _next_observation(self):
        if self.observation_mode == 'Fully Observable':
            return np.concatenate([self.S.plant_pose[self.env_n], self.S.get_franka_angles(self.env_n)])
        elif self.observation_mode == 'Grayscale Image':
            return self.S.grayscale_cam_imgs[self.env_n].flatten()

    def reset(self):
        self.ep_steps = 0
        dof_states = np.zeros(9, dtype=gymapi.DofState.dtype)
        dof_states['pos'] = np.array([-0.2, 0.5, 0.75, -2, 1.25, 2.25, -1, 0, 0]) #np.array([1, 0.5, 0, -0.9425, 0, 1.12, 0, 0, 0])
        
        # self.S.set_franka_angles(np.array([-1, 0.5, 0.75, -2, 1.25, 2.25, -1, 0, 0]), self.env_n)
        self.S.gym.set_actor_dof_states(self.S.envs[self.env_n], self.S.franka_handles[self.env_n], dof_states, gymapi.STATE_ALL)
        self.S.set_franka_angles_target(dof_states['pos'], self.env_n)
        self.S.get_franka_angles(self.env_n)
        self.S.sim_step(skip_images = True)

        print(f'Clearances: {self.clearances}')

        return self._next_observation()
