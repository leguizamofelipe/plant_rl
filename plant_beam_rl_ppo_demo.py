from time import sleep
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from basic_model.plant_beam_model_ppo_env import PlantBeamModelPPOEnvironment

# Environment definition
env = PlantBeamModelPPOEnvironment()

time_steps = 100

model = PPO('MlpPolicy', env, verbose = 1, device = 'cuda')

model.learn(total_timesteps=int(time_steps), n_eval_episodes = 30)

################################ PLOTTING #####################################

fig, axs = plt.subplots(5, sharex = True)

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

plt.show()

print('done')

# ############################## DO A TEST RUN ##############################
env.reset()
count = 0
while(True):
    obs = env._next_observation()
    action = model.predict(obs)[0]
    obs, reward, done, obj = env.step(action)
    title = f'R: {round(reward, 2)} Force: {round(env.force, 2)} Location: {round(env.location, 2)} \n Delta Force: {round(action[0], 2)} Delta Location: {round(action[1], 2)}'

    env.P.plot_plant(save=True, filename=f'replay/step_{count}.png', title=title)
    plt.close()
    if done: break
    count+=1