from math import exp
import gym
from gym import spaces
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from basic_model.plant_beam_model import PlantBeamModel
import random
import tensorflow as tf

Adam = tf.keras.optimizers.Adam(learning_rate = 0.01)

from basic_model.icm_networks import *

class PlantBeamModelContinuousEnvironment(gym.Env):
    """A plant environment for OpenAI gym"""
    metadata = {'render.modes': ['human']} # TODO understand what this does

    def __init__(self, icm = False):
        self.max_fruit_radius = 1
        self.P = PlantBeamModel(random.random()*self.max_fruit_radius)

        self.force = 0
        self.max_ep_len = 20
        self.max_delta_force = 20

        self.batch_size = 10

        self.alpha_prop = []
        self.beta_prop = []
        self.gamma_prop = []
        self.manipulations = []
        self.rewards = []
        self.cumulative_breaks = []

        self.breaks= 0
        self.ep_alpha = 0
        self.ep_beta = 0
        self.ep_gamma = 0
        self.ep_abs_reward = 0
        self.pushes = 0
        self.ep_reward = 0
        self.eps = 0
        self.location = self.P.p_len/2

        self.timesteps = 1

        # Hold last 11 states
        self.states_buffer = []

        # Hold last 10 actions, rewards
        self.actions_buffer = []
        self.rewards_buffer = []

        # delta force, location
        self.action_space = spaces.Box(low = np.array([-self.max_delta_force, -self.P.p_len/20]), high = np.array([self.max_delta_force, self.P.p_len/20]))

        # Continuous observation state: x_plant, y_plant, x_cf, y_cf, r_f, f_app
        self.observation_space = spaces.Box(low = np.concatenate((-10*np.ones(2*self.P.resolution + 4), np.array([0, 0]))), high = np.concatenate((10*np.ones(2*self.P.resolution + 4), np.array([500, self.P.p_len]))))

        ### ICM
        self.icm = icm
        # if icm:
        #     self.forward_model = ForwardModel(self.observation_space.shape)
        #     self.forward_model.compile(optimizer = Adam, loss = 'mse')
        #     self.inverse_model = InverseModel(self.action_space.shape)
        #     self.inverse_model.compile(optimizer = Adam, loss = 'mse')
        #     self.feature_extractor = FeatureExtractor(self.observation_space.shape)
        #     self.feature_extractor.compile(optimizer = Adam, loss = 'mse')            

        if icm:
            self.icm_model = ICMModel(self.observation_space.shape, self.action_space.shape).built_model
            self.icm_model.compile(optimizer=Adam, loss='mse')

    def _take_action(self, action):
        self.force+=action[0]
        self.location+=action[1]

        if self.location <= 0:
            self.location = 0.15
        if self.location >= 3:
            self.location = 3
    
        self.P.apply_force(self.force, self.location)

    def step(self, action):
        done = False
        
        self._take_action(action)
        nu = self.P.calculate_occlusion()
        gamma = 0
        alpha = 0

        # k = sum(abs(self.P.max_von_mises))*10**-10
        k = max(abs(self.P.max_von_mises))*10**-6
        # alpha = -750*k**2
        alpha = max(-3000, -exp(0.5*(k-75)))#-750*k**2
        beta = -100*nu

        break_plant = False
        # Assume that a stress of 90 MPa breaks the plant
        if max(abs(self.P.max_von_mises)) > 75*10**6:
            gamma = 0
            # done = True
            self.breaks+=1

        if max(abs(self.P.max_von_mises)) > 90*10**6:
            break_plant = True
            gamma = -5000

        if nu == 0 and break_plant==False:
            done = True
            gamma = abs(self.ep_reward)+1000

        # if max(self.P.max_von_mises) > 90*10**6:
        #     self.breaks+=1

        self.ep_alpha += alpha
        self.ep_beta += beta
        self.ep_gamma += gamma
        self.ep_abs_reward += abs(alpha) + abs(beta) + abs(gamma)
        self.pushes+=1

        reward = alpha + gamma + beta

        self.ep_reward += reward

        obs = self._next_observation()

        self.states_buffer.append(obs)
        self.actions_buffer.append(action)
        self.rewards_buffer.append(reward)

        if self.timesteps % (self.batch_size+1)==0:
            self.update_icm()

        if self.pushes > self.max_ep_len:
            done = True

        self.timesteps+=1

        return obs, reward, done, {'gamma': gamma, 'beta' : beta, 'occ' : nu, 'alpha' : alpha, 'success' : nu==0, 'break_plant': break_plant}

    def _next_observation(self):
        return np.concatenate((self.P.x, self.P.y, np.array([self.P.calculate_occlusion(), self.P.fruit_y_center, self.P.fruit_x_center, self.P.fruit_radius, self.force, self.location])))
    
    def reset(self, set_occlusion=False):
        self.eps += 1
        if self.eps % 1000 == 0:
            self.P.plot_plant(save=True, filename=f'output/Ep_{self.eps}_final_pose.png', title = f'Reward: {self.ep_reward} \n Broke plant?: {max(self.P.max_von_mises) > 90*10**6}')
        
        self.cumulative_breaks.append(self.breaks)

        self.force = 0
        self.location = self.P.x[int(len(self.P.x)/2)]

        # Randomly set the elastic modulus within a reasonable bound
        self.P.E = random.gauss(10, 1) * 1e9
        self.P.apply_force(0, 1.5)

        self.rewards.append(self.ep_reward)
        self.manipulations.append(self.pushes)
        if self.ep_abs_reward != 0:
            self.alpha_prop.append(self.ep_alpha)#abs(self.ep_alpha)/self.ep_abs_reward)
            self.beta_prop.append(self.ep_beta)#abs(self.ep_beta)/self.ep_abs_reward)
            self.gamma_prop.append(self.ep_gamma)#abs(self.ep_gamma)/self.ep_abs_reward)
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

        self.set_occlusion()

        return self._next_observation()

    def set_occlusion(self):
        # Place the fruit somewhere
        self.P.fruit_radius = random.random()*self.max_fruit_radius
        self.P.fruit_y_center = random.random()-1
        self.P.fruit_x_center = random.random()*2.5+1

        # Try again if there is no occlusion
        if self.P.calculate_occlusion() == 0:
            self.set_occlusion()

    def update_icm(self):
        s_t=np.array(self.states_buffer[:self.batch_size-1])
        s_t1=np.array(self.states_buffer[1:self.batch_size])

        actions=np.array(self.actions_buffer[:self.batch_size-1])
        rewards = np.array(self.rewards_buffer[:self.batch_size-1])
        self.icm_model.train_on_batch([s_t, s_t1, actions, rewards], np.zeros(self.batch_size-1,))

    # def create_models(self):
    #     if self.icm:
    #         self.forward_net = ForwardModel()
    #         self.inverse_net = InverseModel()
    #         self.feature_extractor = FeatureExtractor()

    #         self.forward_net.compile(optimizer=Adam, loss='mse')
    #         self.inverse_net.compile(optimizer=Adam, loss='mse')
    #         self.feature_extractor.compile(optimizer=Adam, loss='mse')

    #     else:
    #         pass

    # def update_models(self):
    #     self.forward_net
    #     self.inverse_net
    #     self.feature_extractor

