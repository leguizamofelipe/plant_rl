import pfrl
import torch
import torch.nn
import gym
import numpy as np
from basic_model.plant_model_env import BasicPlantModelEnvironment
import matplotlib.pyplot as plt

for n_joints in [3, 5, 10, 20, 25, 30]:

    # n_joints = 3

    link_len = 25/n_joints

    # Environment definition
    env = BasicPlantModelEnvironment(link_len, n_joints, randomize=True)

    # Agent definition
    class QFunction(torch.nn.Module):
        def __init__(self, obs_size, n_actions):
            super().__init__()
            self.l1 = torch.nn.Linear(obs_size, 50)
            self.l2 = torch.nn.Linear(50, 50)
            self.l3 = torch.nn.Linear(50, n_actions)

        def forward(self, x):
            h = x
            h = torch.nn.functional.relu(self.l1(h))
            h = torch.nn.functional.relu(self.l2(h))
            h = self.l3(h)
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

    n_episodes = 1000
    max_episode_len = 10

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
        beta = 0
        gamma = 0

        # For each episode
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            if env.P.calculate_occlusion() > 0:
                action = agent.act(obs)
                obs, reward, done, res = env.step(action)
                gamma = res['gamma']
                alpha += res['alpha']
                beta = 0#5
                reward -= beta*n_manipulations
                n_manipulations+=1
                cum_occ+=res['occ']
                # env.P.plot_plant()
                R += reward
                abs_R += gamma + beta + abs(res['alpha'])
                t += 1
                reset = t == max_episode_len
                agent.observe(obs, reward, done, reset)
                if done or reset:
                    break
            else:
                # Occlusion is 0, so regen the plant model to try to get occlusion
                env.reset()
        if i % 10 == 0:
            print('episode:', i, 'R:', R)
        if i % 50 == 0:
            print('statistics:', agent.get_statistics())
        
        alphas.append(abs(alpha)/abs_R if abs_R else 0)
        betas.append(t*abs(beta)/abs_R if abs_R else 0)
        gammas.append(gamma/abs_R if abs_R else 0)
        # alphas.append(abs(alpha))
        # betas.append(t*abs(beta))
        # gammas.append(gamma)

        ep_rewards.append(R)
        ep_manipulations.append(n_manipulations)
        ep_occs.append(cum_occ)
    print('Finished.')

    fig, axs = plt.subplots(6, sharex = True)

    alpha = 0.7

    window = 20
    running_average = []
    ep_rewards = np.array(ep_rewards)
    for ind in range(len(ep_rewards)-window + 1):
        running_average.append(np.mean(ep_rewards[ind:ind+window]))

    for ind in range(window - 1):
        running_average.insert(0, np.nan)

    axs[0].plot(ep_rewards, label = 'Rewards', color = 'r', alpha = 0.1)
    axs[0].plot(running_average, label = 'Running Average', color = 'black', alpha = alpha)
    axs[0].legend()
    axs[1].plot(ep_manipulations, label = 'Manipulations', color = 'g', alpha = alpha)
    axs[1].legend()
    axs[2].plot(ep_occs, label = 'Occlusion', color = 'b', alpha = alpha)
    axs[2].legend()
    axs[3].plot(alphas, label = 'Strain Contribution (Alpha)', color = 'orange', alpha = alpha)
    axs[3].legend()
    axs[4].plot(betas, label = 'Manipulation Contribution (Beta)', color = 'purple', alpha = alpha)
    axs[4].legend()
    axs[5].plot(gammas, label = 'Success Contribution (Gamma)', color = 'gray', alpha = alpha)
    axs[5].legend()

    # plt.title('Rewards')
    plt.tight_layout()
    # plt.title(f'n = {n_joints} Joints')
    plt.xlabel('Episode')
    # plt.show()
    plt.savefig(f'output/{n_joints}_joints.png')
    plt.close()

    env.P.plot_plant(save=True, tag = f'{n_joints}_final_pose', title = str(env.sigmas))
