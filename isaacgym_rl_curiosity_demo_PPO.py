import torch
import torch.nn as nn
import numpy as np
from isaacgym_sim.isaacgym_env import IsaacGymPlantEnv

import functools

import gym

import pfrl
from pfrl import experiments, utils
from pfrl.agents import PPO

num_envs = 1

outdir = experiments.prepare_output_dir('out')

def make_env():
    env = IsaacGymPlantEnv()
    # Cast observations to float32 because our model uses float32
    env = pfrl.wrappers.CastObservationToFloat32(env)
    return env

def make_batch_env(test):
    return pfrl.envs.MultiprocessVectorEnv(
        [
            functools.partial(make_env, idx, test)
            for idx, env in enumerate(range(num_envs))
        ]
    )

# Only for getting timesteps, and obs-action spaces
sample_env = IsaacGymPlantEnv()
timestep_limit = sample_env.spec.max_episode_steps
obs_space = sample_env.observation_space
action_space = sample_env.action_space
print("Observation space:", obs_space)
print("Action space:", action_space)

assert isinstance(action_space, gym.spaces.Box)

# Normalize observations based on their empirical mean and variance
obs_normalizer = pfrl.nn.EmpiricalNormalization(
    obs_space.low.size, clip_threshold=5
)

obs_size = obs_space.low.size
action_size = action_space.low.size
policy = torch.nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, action_size),
    pfrl.policies.GaussianHeadWithStateIndependentCovariance(
        action_size=action_size,
        var_type="diagonal",
        var_func=lambda x: torch.exp(2 * x),  # Parameterize log std
        var_param_init=0,  # log std = 0 => std = 1
    ),
)

vf = torch.nn.Sequential(
    nn.Linear(obs_size, 64),
    nn.Tanh(),
    nn.Linear(64, 64),
    nn.Tanh(),
    nn.Linear(64, 1),
)

# While the original paper initialized weights by normal distribution,
# we use orthogonal initialization as the latest openai/baselines does.
def ortho_init(layer, gain):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.zeros_(layer.bias)

ortho_init(policy[0], gain=1)
ortho_init(policy[2], gain=1)
ortho_init(policy[4], gain=1e-2)
ortho_init(vf[0], gain=1)
ortho_init(vf[2], gain=1)
ortho_init(vf[4], gain=1)

# Combine a policy and a value function into a single model
model = pfrl.nn.Branched(policy, vf)

opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)

update_interval = 2048
batch_size = 64
epochs = 10
steps = 10e6

agent = PPO(
    model,
    opt,
    obs_normalizer=obs_normalizer,
    gpu=True,
    update_interval=update_interval,
    minibatch_size=batch_size,
    epochs=epochs,
    clip_eps_vf=None,
    entropy_coef=0,
    standardize_advantages=True,
    gamma=0.995,
    lambd=0.97,
)

# if load or load_pretrained:
#     # either load or load_pretrained must be false
#     assert not load or not load_pretrained
#     if load:
#         agent.load(load)
#     else:
#         agent.load(utils.download_model("PPO", env, model_type="final")[0])


experiments.train_agent_batch_with_evaluation(
    agent=agent,
    env=make_batch_env(False),
    eval_env=make_batch_env(True),
    outdir=outdir,
    steps=steps,
    eval_n_steps=None,
    eval_n_episodes=100,
    eval_interval=100000,
    log_interval=1000,
    max_episode_len=timestep_limit,
    save_best_so_far_agent=False,
)
