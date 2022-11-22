from isaacgym_sim.isaacgym_env import IsaacGymPlantEnv

import argparse
import os

# Prevent numpy from using multiple threads
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # NOQA:E402
from torch import nn  # NOQA:E402

import pfrl  # NOQA:E402
from pfrl import experiments, utils  # NOQA:E402
from pfrl.agents import a3c  # NOQA:E402
from pfrl.optimizers import SharedRMSpropEpsInsideSqrt  # NOQA:E402
from pfrl.policies import SoftmaxCategoricalHead  # NOQA:E402
from pfrl.wrappers import atari_wrappers  # NOQA:E402

env = IsaacGymPlantEnv()
obs_size = env.observation_space.shape[0]
n_actions = env.action_space.n

model = nn.Sequential(
    nn.Conv2d(obs_size, 16, 8, stride=4),
    nn.ReLU(),
    nn.Conv2d(16, 32, 4, stride=2),
    nn.ReLU(),
    nn.Flatten(),
    nn.Linear(2592, 256),
    nn.ReLU(),
    pfrl.nn.Branched(
        nn.Sequential(
            nn.Linear(256, n_actions),
            SoftmaxCategoricalHead(),
        ),
        nn.Linear(256, 1),
    ),
)

opt = SharedRMSpropEpsInsideSqrt(model.parameters(), lr=7e-4, eps=1e-1, alpha=0.99)

def phi(x):
        # Feature extractor
        return np.asarray(x, dtype=np.float32) / 255

steps = 8*10e7
lr = 7e-4

agent = a3c.A3C(
    model,
    opt,
    t_max=5,
    gamma=0.99,
    beta=1e-2,
    phi=phi,
    max_grad_norm=40.0,
)

def make_env(process_idx, test):

        return IsaacGymPlantEnv()

def lr_setter(env, agent, value):
            for pg in agent.optimizer.param_groups:
                assert "lr" in pg
                pg["lr"] = value

lr_decay_hook = experiments.LinearInterpolationHook(
            steps, lr, 0, lr_setter
        )

experiments.train_agent_async(
    agent=agent,
    outdir='out',
    processes=1,
    make_env=make_env,
    profile=False,
    steps=steps,
    eval_n_steps=150000,
    eval_n_episodes=None,
    eval_interval=250000,
    global_step_hooks=[lr_decay_hook],
    save_best_so_far_agent=True,
)