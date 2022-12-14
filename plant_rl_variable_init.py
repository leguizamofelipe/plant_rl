import pfrl
import torch
import torch.nn
import gym
import numpy as np
from basic_model.plant_model_env import BasicPlantModelEnvironment
import matplotlib.pyplot as plt

for n_joints in [7]:

    # n_joints = 3

    link_len = 25/n_joints

    # Environment definition
    env = BasicPlantModelEnvironment(link_len, n_joints, randomize=True)

    # Agent definition
    class QFunction(torch.nn.Module):
        def __init__(self, obs_size, n_actions):
            super().__init__()
            self.l1 = torch.nn.Linear(obs_size, 100)
            self.l2 = torch.nn.Linear(100, 100)
            self.l3 = torch.nn.Linear(100, n_actions)

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

    n_episodes = 2000
    max_episode_len = 20

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

        valid = False

        # Check that the randomly generated plant is occluded
        while not valid:
            if env.P.calculate_occlusion() > 0:
                valid = True
                if i % 200 == 0:
                    env.P.plot_plant(save = True, tag = f'ep-{i}-init')
            else:
                # Occlusion is 0, so regen the plant model to try to get occlusion
                env.reset()

        # For each episode
        while True:
            # Uncomment to watch the behavior in a GUI window
            # env.render()
            action = agent.act(obs)
            obs, reward, done, res = env.step(action)
            gamma = res['gamma']
            alpha += res['alpha']
            beta = 0
            reward -= beta*n_manipulations
            n_manipulations+=1
            cum_occ+=res['occ']
            # env.P.plot_plant()
            R += reward
            abs_R += gamma + beta + abs(res['alpha'])
            t += 1
            reset = t == max_episode_len
            agent.observe(obs, reward, done, reset)
            if done:
                break
            if reset:
                break
        if i % 10 == 0:
            print('episode:', i, 'R:', R)
        if i % 200 == 0:
            print('statistics:', agent.get_statistics())
            env.P.plot_plant(save = True, tag = f'ep-{i}-final', title = f'Manipulations: {n_manipulations}\n {str(env.sigmas)}')
        
        alphas.append(alpha)#/abs_R if abs_R else 0)
        betas.append(t*abs(beta)/abs_R if abs_R else 0)
        gammas.append(gamma)#/abs_R if abs_R else 0)
        # alphas.append(abs(alpha))
        # betas.append(t*abs(beta))
        # gammas.append(gamma)

        ep_rewards.append(R)
        ep_manipulations.append(n_manipulations)
        ep_occs.append(cum_occ)
    print('Finished.')

    fig, axs = plt.subplots(3, sharex = True)

    alpha = 0.7

    window = 20
    running_average = []
    ep_rewards = np.array(ep_rewards)
    for ind in range(len(ep_rewards)-window + 1):
        running_average.append(np.mean(ep_rewards[ind:ind+window]))

    for ind in range(window - 1):
        running_average.insert(0, np.nan)

    axs[0].plot(ep_rewards, label = 'Rewards', color = 'r', alpha = 0.1)
    axs[0].plot(running_average, label = 'Running Average', color = 'r', alpha = alpha)
    axs[0].legend()
    # axs[1].plot(ep_manipulations, label = 'Manipulations', color = 'g', alpha = alpha)
    # axs[1].legend()
    # axs[1].plot(ep_occs, label = 'Cumulative Occlusion', color = 'b', alpha = alpha)
    # axs[1].legend()
    # axs[2].plot(np.array(alphas)+np.array(betas)+np.array(gammas), label = 'A+B+Ga', color = 'b', alpha = alpha)
    # axs[2].legend()
    axs[1].plot(alphas, label = 'Strain Contribution (Alpha)', color = 'orange', alpha = alpha)
    axs[1].legend()
    # axs[3].plot(betas, label = 'Manipulation Contribution (Beta)', color = 'purple', alpha = alpha)
    # axs[3].legend()
    axs[2].plot(gammas, label = 'Success Contribution (Gamma)', color = 'gray', alpha = alpha)
    axs[2].legend()

    # plt.title('Rewards')
    plt.tight_layout()
    # plt.title(f'n = {n_joints} Joints')
    plt.xlabel('Episode')
    # plt.show()
    plt.savefig(f'output/{n_joints}_joints.png', dpi=500)
    plt.close()

