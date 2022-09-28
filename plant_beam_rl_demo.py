import pfrl
import torch
import torch.nn
import gym
import numpy as np
import matplotlib.pyplot as plt
from basic_model.plant_beam_model_env import PlantBeamModelEnvironment
from pfrl.policies import SoftmaxCategoricalHead

# Environment definition
env = PlantBeamModelEnvironment()

obs_size = env.observation_space.low.size
n_actions = env.action_space.n
hidden_size = 50

# Agent definition
model = torch.nn.Sequential(
    torch.nn.Linear(obs_size, hidden_size),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_size, hidden_size),
    torch.nn.ReLU(),
    pfrl.nn.Branched(
        torch.nn.Sequential( # pi
            torch.nn.Linear(hidden_size, n_actions),
            SoftmaxCategoricalHead(),
        ),
        torch.nn.Linear(hidden_size, 1), # v
    ),
    )
    
optimizer = torch.optim.Adam(model.parameters(), eps=1e-2)

gamma_rl = 0.9
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.1, random_action_func=env.action_space.sample)
phi = lambda x: x.astype(np.float32, copy=False)
update_interval = 4096

agent = pfrl.agents.PPO(
    model,
    optimizer,
    gamma=0.99,
    gpu=0,
    phi=phi,
    update_interval=update_interval,
    minibatch_size=64,
    epochs=10,
    clip_eps=0.2,
    clip_eps_vf=None,
    standardize_advantages=True,
    entropy_coef=0,
    max_grad_norm=0.5,
)

n_episodes = 2500
max_episode_len = 15

ep_rewards = []
ep_manipulations = []
ep_occs = []

alphas = []
betas = []
gammas = []
for i in range(1, n_episodes + 1):
    obs = env.reset()

    R = 0  # return (sum of rewards)
    abs_R = 0
    t = 0  # time step
    cum_occ = 0
    n_manipulations = 0

    alpha = 0
    gamma = 0

    # For each episode
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        obs, reward, done, res = env.step(action)
        gamma = res['gamma']
        alpha += res['alpha']
        cum_occ+=res['occ']
        # env.P.plot_plant()
        R += reward
        abs_R += gamma + abs(res['alpha'])
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            break
    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())

    alphas.append(abs(alpha)/abs_R)
    gammas.append(gamma/abs_R)
    # alphas.append(abs(alpha))
    # gammas.append(gamma)

    ep_rewards.append(R)
    ep_manipulations.append(n_manipulations)
    ep_occs.append(cum_occ)
print('Finished.')

# fig, axs = plt.subplots(6, sharex = True)

# alpha = 0.7

# axs[0].plot(ep_rewards, label = 'Rewards', color = 'r', alpha = alpha)
# axs[0].legend()
# axs[1].plot(ep_manipulations, label = 'Manipulations', color = 'g', alpha = alpha)
# axs[1].legend()
# axs[2].plot(ep_occs, label = 'Occlusion', color = 'b', alpha = alpha)
# axs[2].legend()
# axs[3].plot(alphas, label = 'Strain Contribution (Alpha)', color = 'orange', alpha = alpha)
# axs[3].legend()
# axs[4].plot(betas, label = 'Manipulation Contribution (Beta)', color = 'purple', alpha = alpha)
# axs[4].legend()
# axs[5].plot(gammas, label = 'Success Contribution (Gamma)', color = 'gray', alpha = alpha)
# axs[5].legend()

# # plt.title('Rewards')
# plt.tight_layout()
# # plt.title(f'n = {n_joints} Joints')
# plt.xlabel('Episode')
# # plt.show()
# plt.savefig(f'output/{n_joints}_joints.png')
# plt.close()

# env.P.plot_plant()
