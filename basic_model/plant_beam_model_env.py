import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from basic_model.plant_beam_model import PlantBeamModel
import random


class PlantBeamModelEnvironment(gym.Env):
    """A plant environment for OpenAI gym"""
    metadata = {'render.modes': ['human']} # TODO understand what this does

    def __init__(self):
        self.P = PlantBeamModel()

        self.force = 0
        
        self.action_space = spaces.Discrete(2)

        # Continuous observation state: x_plant, y_plant, x_cf, y_cf, r_f
        self.observation_space = spaces.Box(low = -10*np.ones(2*self.P.resolution + 3), high = 10*np.ones(2*self.P.resolution + 3))

    def _take_action(self, action):
        if action == 0:
            self.force+=10
            self.P.apply_force(self.force, 1.5)
            return False
        elif action == 1:
            return True

    def step(self, action):
        done = self._take_action(action)
        nu = self.P.calculate_occlusion()

        alpha = -sum(abs(self.P.max_von_mises))*10**-9
        gamma = 0
        if nu == 0:
            done = True
            gamma = 500

        reward = alpha + gamma

        obs = self._next_observation()

        return obs, reward, done, {'gamma': gamma, 'occ' : nu, 'alpha' : alpha}

    def _next_observation(self):
        return np.concatenate([self.P.x, self.P.y, np.array([self.P.fruit_x_pos, self.P.fruit_y_pos, self.P.fruit_radius])])

    def reset(self):
        return self._next_observation()
