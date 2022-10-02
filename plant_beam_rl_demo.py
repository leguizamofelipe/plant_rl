import pfrl
import torch
import torch.nn
import gym
import numpy as np
import matplotlib.pyplot as plt

from basic_model.plant_beam_model_ppo_env import PlantBeamModelPPOEnvironment

# Environment definition
env = PlantBeamModelPPOEnvironment()

# Agent definition
class QFunction(torch.nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.l1 = torch.nn.Linear(obs_size, 100)
        self.l2 = torch.nn.Linear(100, 100)
        # self.l3 = torch.nn.Linear(100, 100)
        # self.l4 = torch.nn.Linear(100, 100)
        self.l5 = torch.nn.Linear(100, n_actions)

    def forward(self, x):
        h = x
        h = torch.nn.functional.relu(self.l1(h))
        h = torch.nn.functional.relu(self.l2(h))
        # h = torch.nn.functional.relu(self.l3(h))
        # h = torch.nn.functional.relu(self.l4(h))
        h = self.l5(h)
        return pfrl.action_value.DiscreteActionValue(h)

# Start up the Q function
obs_size = env.observation_space.low.size
n_actions = env.action_space.n
q_func = QFunction(obs_size, n_actions)

optimizer = torch.optim.Adam(q_func.parameters(), eps=1e-2)

gamma_rl = 0.9
replay_buffer = pfrl.replay_buffers.ReplayBuffer(capacity=10 ** 6)
explorer = pfrl.explorers.ConstantEpsilonGreedy(epsilon=0.3, random_action_func=env.action_space.sample)
phi = lambda x: x.astype(np.float32, copy=False)

# Now create an agent that will interact with the environment.
agent = pfrl.agents.DoubleDQN(
    q_func,
    optimizer,
    replay_buffer,
    gamma_rl,
    explorer,
    replay_start_size=500,
    update_interval=1,
    target_update_interval=100,
    phi=phi,
)

n_episodes = 2000
max_episode_len = 30

alpha_prop = []
beta_prop = []
gamma_prop = []
manipulations = []
rewards = []

for i in range(1, n_episodes + 1):
    obs = env.reset()
    t = 0
    R = 0  # return (sum of rewards)
    ep_alpha = 0
    ep_beta = 0
    ep_gamma = 0
    ep_abs_reward = 0
    pushes = 0
    # For each episode
    while True:
        # Uncomment to watch the behavior in a GUI window
        # env.render()
        action = agent.act(obs)
        pushes += action
        obs, reward, done, res = env.step(action)

        ######################### INSTRUMENTATION #####################
        ep_alpha += res['alpha']
        ep_beta += res['beta']
        ep_gamma += res['gamma']
        ep_abs_reward += abs(res['alpha']) + abs(res['beta']) + abs(res['gamma'])
        ######################### INSTRUMENTATION #####################
        
        # reward -= t*50
        R += reward
        t += 1
        reset = t == max_episode_len
        agent.observe(obs, reward, done, reset)
        if done or reset:
            if i % int(n_episodes/20) == 0:
                env.P.plot_plant(save=True, filename=f'Ep_{i}_final_pose.png', title = f'Reward: {R}')
            break

    #################### INSTRUMENTATION ##################
    rewards.append(R)
    manipulations.append(pushes)
    if ep_abs_reward != 0:
        alpha_prop.append(abs(ep_alpha)/ep_abs_reward)
        beta_prop.append(abs(ep_beta)/ep_abs_reward)
        gamma_prop.append(abs(ep_gamma)/ep_abs_reward)
    else:
        alpha_prop.append(0)
        beta_prop.append(0)
        gamma_prop.append(0)
    #######################################################

    if i % 10 == 0:
        print('episode:', i, 'R:', R)
    if i % 50 == 0:
        print('statistics:', agent.get_statistics())

fig, axs = plt.subplots(6, sharex = True)

alpha = 0.7
window = 20
running_average = []
ep_rewards = np.array(rewards)
for ind in range(len(ep_rewards)-window + 1):
    running_average.append(np.mean(ep_rewards[ind:ind+window]))

for ind in range(window - 1):
    running_average.insert(0, np.nan)

axs[0].plot(rewards, label = 'Rewards', color = 'r', alpha = 0.1)
axs[0].plot(running_average, color = 'r', alpha = alpha)
axs[0].legend()
axs[1].plot(manipulations, label = 'Manipulations', color = 'g', alpha = alpha)
axs[1].legend()
# axs[2].plot(ep_occs, label = 'Occlusion', color = 'b', alpha = alpha)
# axs[2].legend()
axs[3].plot(alpha_prop, label = 'Strain Contribution (Alpha)', color = 'orange', alpha = alpha)
axs[3].legend()
axs[4].plot(beta_prop, label = 'Occlusion Contribution (Beta)', color = 'purple', alpha = alpha)
axs[4].legend()
axs[5].plot(gamma_prop, label = 'Success Contribution (Gamma)', color = 'gray', alpha = alpha)
axs[5].legend()

# plt.title('Rewards')
plt.tight_layout()
# plt.title(f'n = {n_joints} Joints')
plt.xlabel('Episode')
plt.show()
# plt.savefig(f'output/{n_joints}_joints.png')
