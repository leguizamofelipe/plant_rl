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
    metadata = {'render.modes': ['human']} 

    def __init__(self):
        self.P = PlantBeamModel()

        self.force = 0
        
        self.action_space = spaces.Discrete(10)

        # Continuous observation state: x_plant, y_plant, x_cf, y_cf, r_f, f_app
        self.observation_space = spaces.Box(low = np.concatenate((-10*np.ones(2*self.P.resolution + 3), np.array([0]))), high = np.concatenate((10*np.ones(2*self.P.resolution + 3), np.array([500]))))
        
        # f_app
        # self.observation_space = spaces.Box(low = np.array([0]), high = np.array([500]))

    def _take_action(self, action):
        if action == 1:
            # self.force+=10
            self.P.apply_force(self.force, 1.5)
            return False
        elif action == 0:
            return True
            # return False

    def step(self, action):
        self.force+=10
        self.P.apply_force(self.force, 1.5)
        done = self._take_action(action)
        nu = self.P.calculate_occlusion()
        gamma = 0
        alpha = 0

        if done and nu>0:
            gamma = -500
        elif nu == 0:
            done = True
            gamma = 500

        k = sum(abs(self.P.max_von_mises))*10**-10
        alpha = -k**2
        beta = -30*nu

        reward = alpha + gamma + beta

        obs = self._next_observation()

        return obs, reward, done, {'gamma': gamma, 'beta' : beta, 'occ' : nu, 'alpha' : alpha}

    def _next_observation(self):
        return np.concatenate([self.P.x, self.P.y, np.array([self.P.fruit_y_center, self.P.fruit_x_center, self.P.fruit_radius, self.force])])
        # return np.array([self.force])

    def reset(self):
        self.force = 0
        self.P.apply_force(0, 1.5)
        return self._next_observation()
