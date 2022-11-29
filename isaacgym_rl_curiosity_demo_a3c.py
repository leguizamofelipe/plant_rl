import os
from isaacgym_sim import Simulation

os.environ['OMP_NUM_THREADS'] = '1'

simulation = Simulation()

from parallel_env import ParallelEnv
import torch.multiprocessing as mp

n_envs = simulation.num_envs
n_actions = 14
n_obs = 4899

input_shape = [n_obs] 

mp.set_start_method('spawn')

parallel_env = ParallelEnv(n_envs=n_envs, n_actions=n_actions, simulation=simulation, input_shape=input_shape, icm=True)