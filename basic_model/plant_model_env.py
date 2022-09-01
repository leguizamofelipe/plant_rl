import gym
from gym import spaces
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from basic_model.plant_model import PlantModel

class BasicPlantModelEnvironment(gym.Env):
    """A plant environment for OpenAI gym"""
    metadata = {'render.modes': ['human']} # TODO understand what this does

    def __init__(self):
        self.P = PlantModel(10)
        # Action space is angle followed by joint index
        # self.action_space = spaces.Box(low = np.array([0, 0]), high = np.array([4, 360]), dtype=np.float32)
        
        # Start 
        self.action_space = spaces.Discrete(10)
        self.observation_space = spaces.Box(low = np.array([0, 0, 0, 0, 0, 0]), high = np.array([self.P.max_occlusion, 360, 360, 360, 360, 360]))
        self.initialize()

    def _take_action(self, action):
        action -= 5
        if np.sign(action) != -1:
            action+=1

        theta = 5
        curr_angle = self.P.get_angles()[abs(action)-1]
        
        self.P.rotate_node(abs(action)-1, curr_angle + np.sign(action)*theta)

    def step(self, action):
        self._take_action(action)
        
        occ_factor = self.P.calculate_occlusion()

        if occ_factor == 0:
            done = True
            reward = 100
            obs = self._next_observation()
        else:           
            done = False 
            reward = -occ_factor
            obs = self._next_observation()

        return obs, reward, done, occ_factor

    def _next_observation(self):
        return np.concatenate((np.array([self.P.calculate_occlusion()]), self.P.get_angles()))

    def reset(self):
        self.initialize()

        return self._next_observation()

    def plot_rewards(self, added_title):
        pass

    def initialize(self):
        self.P.rotate_node(0, 45)
        self.P.rotate_node(1, 0)
        self.P.rotate_node(2, 0)
        self.P.rotate_node(3, 0)
        self.P.rotate_node(4, 0)
        self.P.rotate_node(5, 0)