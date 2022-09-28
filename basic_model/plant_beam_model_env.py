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
        
        # Max force to apply
        P_max = 200

        # a = [x_app, P]
        self.action_space = spaces.Box(low = np.array([0, 0]), high = np.array([self.P.p_len, P_max]))

        # Continuous observation state: max Von Mises stress, occlusion
        self.observation_space = spaces.Box(low = np.array([0, 0]), high = np.array([10e31, 1]))

    def _take_action(self, action):
        P_des = action[1]
        x_des = action[0]
        
        self.P.apply_force(P_des, x_des)

    def step(self, action):
        self._take_action(action)
        
        occ_factor = self.P.calculate_occlusion()
        alpha = 0
        if max(abs(self.sigmas) > 3):
            alpha = 0 #-100 #-sum(np.abs(self.sigmas) ** 3 )
        # alpha = 0
        if occ_factor == 0:
            done = True
            gamma = 250
            obs = self._next_observation()
        else:           
            done = False 
            # reward = -occ_factor
            gamma = 0
            obs = self._next_observation()

        reward = alpha + gamma

        return obs, reward, done, {'gamma':gamma, 'occ' : occ_factor, 'alpha' : alpha}

    def _next_observation(self):
        return np.array([self.P.max_von_mises, self.P.return_occlusion()])

    def reset(self):
        return self._next_observation()
