import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from basic_model.plant_beam_model import PlantBeamModel
import random


class PlantBeamModelPPOEnvironment(gym.Env):
    """A plant environment for OpenAI gym"""
    metadata = {'render.modes': ['human']} # TODO understand what this does

    def __init__(self):
        self.P = PlantBeamModel()

        self.force = 0
        self.max_ep_len = 20
        self.max_delta_force = 20

        self.alpha_prop = []
        self.beta_prop = []
        self.gamma_prop = []
        self.manipulations = []
        self.rewards = []
        self.ep_alpha = 0
        self.ep_beta = 0
        self.ep_gamma = 0
        self.ep_abs_reward = 0
        self.pushes = 0
        self.ep_reward = 0
        self.eps = 0
        self.location = self.P.p_len/2

        # delta force, location
        self.action_space = spaces.Box(low = np.array([10, -int(self.P.p_len/20)]), high = np.array([self.max_delta_force, int(self.P.p_len/20)]))

        # Continuous observation state: x_plant, y_plant, x_cf, y_cf, r_f, f_app
        self.observation_space = spaces.Box(low = np.concatenate((-10*np.ones(2*self.P.resolution + 4), np.array([0]))), high = np.concatenate((10*np.ones(2*self.P.resolution + 4), np.array([500]))))
        

    def _take_action(self, action):
        self.force+=action[0]
        self.location+=action[1]
        self.P.apply_force(self.force, self.location)

    def step(self, action):
        done = False
        self._take_action(action)
        nu = self.P.calculate_occlusion()
        gamma = 0
        alpha = 0

        if nu == 0:
            done = True
            gamma = 500

        k = sum(abs(self.P.max_von_mises))*10**-10
        alpha = -750*k**2
        beta = -100*nu

        self.ep_alpha += alpha
        self.ep_beta += beta
        self.ep_gamma += gamma
        self.ep_abs_reward += abs(alpha) + abs(beta) + abs(gamma)
        self.pushes+=1

        reward = alpha + gamma + beta

        self.ep_reward += reward

        obs = self._next_observation()

        if self.pushes > self.max_ep_len:
            done = True

        return obs, reward, done, {'gamma': gamma, 'beta' : beta, 'occ' : nu, 'alpha' : alpha}

    def _next_observation(self):
        return np.concatenate((self.P.x, self.P.y, np.array([self.P.calculate_occlusion(), self.P.fruit_x_pos, self.P.fruit_y_pos, self.P.fruit_radius, self.force])))
    
    def reset(self):
        self.eps += 1
        if self.eps % 1000 == 0:
            self.P.plot_plant(save=True, filename=f'Ep_{self.eps}_final_pose.png', title = f'Reward: {self.ep_reward}')

        self.force = 0
        self.location = self.P.x[int(len(self.P.x)/2)]
        self.P.apply_force(0, 1.5)

        self.rewards.append(self.ep_reward)
        self.manipulations.append(self.pushes)
        if self.ep_abs_reward != 0:
            self.alpha_prop.append(abs(self.ep_alpha)/self.ep_abs_reward)
            self.beta_prop.append(abs(self.ep_beta)/self.ep_abs_reward)
            self.gamma_prop.append(abs(self.ep_gamma)/self.ep_abs_reward)
        else:
            self.alpha_prop.append(0)
            self.beta_prop.append(0)
            self.gamma_prop.append(0)
        
        self.ep_alpha = 0
        self.ep_beta = 0
        self.ep_gamma = 0
        self.ep_abs_reward = 0
        self.pushes = 0
        self.ep_reward = 0

        return self._next_observation()
