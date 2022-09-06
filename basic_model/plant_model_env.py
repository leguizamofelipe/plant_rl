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

    def __init__(self, link_len, n_joints):
        self.P = PlantModel(10, link_len, n_joints)
        # Action space is angle followed by joint index
        # self.action_space = spaces.Box(low = np.array([0, 0]), high = np.array([4, 360]), dtype=np.float32)
        
        # Start 
        self.action_space = spaces.Discrete(self.P.total_joints*2)
        high = np.concatenate((np.array([self.P.max_occlusion]), np.zeros(self.P.total_joints)))
        self.observation_space = spaces.Box(low = np.zeros(self.P.total_joints+1), high = high)
        self.sigmas = np.zeros(self.P.total_joints)

        self.initialize()

    def _take_action(self, action):
        action -= self.P.total_joints
        if np.sign(action) != -1:
            action+=1

        theta = 5
        curr_angle = self.P.get_angles()[abs(action)-1]
        
        self.P.rotate_node(abs(action)-1, curr_angle + np.sign(action)*theta)

        self.sigmas[abs(action)-1] += np.sign(action)

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
        return np.concatenate((np.array([self.P.calculate_occlusion()]), self.P.get_angles()))

    def reset(self):
        self.initialize()

        return self._next_observation()

    def plot_rewards(self, added_title):
        pass

    def initialize(self):
        self.P.rotate_node(0, 45)
        for i in range(1, self.P.total_joints+1): self.P.rotate_node(i, 0)
        self.sigmas = np.zeros(self.P.total_joints)
