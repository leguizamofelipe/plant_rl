import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from basic_model.plant_beam_model_ppo_env import PlantBeamModelPPOEnvironment
import shutil
import os

model = SAC.load('pls.zip')
env = PlantBeamModelPPOEnvironment()
obs = env.reset()

breaks = 0
success = 0
save_dir = f'output/forces'
for file in os.listdir(save_dir):
    os.remove(os.path.join(save_dir, file))

for c in range(0, 25):
    ep_force = []
    ep_rewards = []
    delta_x = []
    R = 0
    for i in range(0, 20):
        if env.P.calculate_occlusion() == 0:
            success+=1
            break

        action = model.predict(obs)[0]
        
        ep_force.append(env.force)
        delta_x.append(action[1])

        obs, reward, done, res = env.step(action)
        ep_rewards.append(R)
        R+=reward

        if res['break_plant']: 
            breaks+=1
            break

    if env.P.calculate_occlusion()>0:#res['break_plant']: 
        fig, ax = plt.subplots(2, sharex = True)
        ax[0].set_title(f'Broke Plant?: {res["break_plant"]} \n Ep Reward: {R} \n Max sigma: {round(max(env.P.max_von_mises)*10**-6, 2)}')
        ax[0].plot(ep_force)
        ax[0].plot(delta_x)
        ax[1].plot(ep_rewards)
        plt.tight_layout()
        plt.savefig(f'output/forces/force_{c}.png', dpi = 200)
        plt.close()
        env.P.plot_plant(save=True, filename=f'forces/plant_{c}.png')
        plt.close()
    plt.close()
    obs = env.reset()

print(f'Breaks: {breaks}')
print(f'Success: {success}')
