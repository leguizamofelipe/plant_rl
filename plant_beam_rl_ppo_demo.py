import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from basic_model.plant_beam_model_ppo_env import PlantBeamModelPPOEnvironment

cwd = os.getcwd()
for file in os.listdir('output'):
    if 'final_pose' in file:
        os.remove(os.path.join(cwd, 'output', file))

# Environment definition
env = PlantBeamModelPPOEnvironment()

time_steps = 2000

model = PPO('MlpPolicy', env, verbose = 1, device = 'cuda')

model.learn(total_timesteps=int(time_steps), n_eval_episodes = 30)

################################ PLOTTING #####################################

fig, axs = plt.subplots(6, sharex = True)

alpha = 0.7
window = 20
running_average = []
ep_rewards = np.array(env.rewards)
for ind in range(len(ep_rewards)-window + 1):
    running_average.append(np.mean(ep_rewards[ind:ind+window]))

for ind in range(window - 1):
    running_average.insert(0, np.nan)

axs[0].plot(env.rewards, label = 'Rewards', color = 'r', alpha = 0.1)
axs[0].plot(running_average, color = 'r', alpha = alpha)
axs[0].legend()
axs[1].plot(env.manipulations, label = 'Manipulations', color = 'g', alpha = alpha)
axs[1].legend()
axs[2].plot(env.alpha_prop, label = 'Strain Contribution (Alpha)', color = 'orange', alpha = alpha)
axs[2].legend()
axs[3].plot(env.beta_prop, label = 'Occlusion Contribution (Beta)', color = 'purple', alpha = alpha)
axs[3].legend()
axs[4].plot(env.gamma_prop, label = 'Success Contribution (Gamma)', color = 'gray', alpha = alpha)
axs[4].legend()
axs[5].plot(env.cumulative_breaks, label = 'Cumulative plant breaks', color = 'black', alpha = alpha)
axs[5].legend()
plt.savefig('output/curves.png')
plt.show()

print('done')

# ############################## DO AN EVALUATION ##############################
# Clear out past data
for file in os.listdir('output/replay'):
    os.remove(os.path.join(cwd, 'output/replay', file))

n_tries = 100

success = np.zeros(n_tries)
breaks = np.zeros(n_tries)

for i in range(0, n_tries):
    env.reset()
    count = 0
    while(True):
        obs = env._next_observation()
        action = model.predict(obs)[0]
        obs, reward, done, obj = env.step(action)
        
        if obj['success']: success[i] += 1 
        if obj['break_plant']: breaks[i] += 1

        if i==0:
            title = f'R: {round(reward, 2)} Force: {round(env.force, 2)} Location: {round(env.location, 2)} \n Delta Force: {round(action[0], 2)} Delta Location: {round(action[1], 2)} \n Episode Reward: {round(env.ep_reward, 2)}'

            env.P.plot_plant(save=True, filename=f'replay/step_{count}.png', title=title)
            plt.close()
        
        if done: break
        count+=1

print(f'Breaks: {sum(breaks)}')
print(f'Success: {sum(success)}')
