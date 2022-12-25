import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC

from basic_model.plant_beam_model_continuous_env import PlantBeamModelContinuousEnvironment
import shutil
import os

model = SAC.load('SAC-500000-randomized-timesteps.zip')
env = PlantBeamModelContinuousEnvironment()
obs = env.reset()

breaks = 0
success = 0
successful_stops = 0
failed_stops = 0

save_dir = f'output/episodes'
for files in os.listdir(save_dir):
    path = os.path.join(save_dir, files)
    try:
        shutil.rmtree(path)
    except OSError:
        os.remove(path)
 
for c in range(0, 100):
    trial_dir = os.path.join(save_dir, f'trial_{c}')
    os.makedirs(trial_dir)
    ep_force = []
    ep_rewards = []
    ep_locations = []
    R = 0
    for i in range(0, 50):
        if env.P.calculate_occlusion() == 0:
            success+=1
            break

        action = model.predict(obs)[0]
        
        ep_force.append(env.force)
        ep_locations.append(env.location)

        obs, reward, done, res = env.step(action)
        ep_rewards.append(R)
        R+=reward

        env.P.plot_plant(save=True, title = f'Broke Plant?: {res["break_plant"]} \n Force: {round(env.force, 3)}\n Max sigma: {round(max(env.P.max_von_mises)*10**-6, 2)}', filename=os.path.join(trial_dir, f'step_{i}.png'))
        plt.close()

        if res['break_plant']: 
            breaks+=1
            break

    if True:#res['break_plant']: 
        fig, ax = plt.subplots(2, sharex = True)
        ax[0].set_title(f'Broke Plant?: {res["break_plant"]} \n Ep Reward: {R} \n Max sigma: {round(max(env.P.max_von_mises)*10**-6, 2)}')
        ax[0].plot(ep_force)
        # ax[0].plot(delta_x)
        ax[1].plot(ep_locations)
        # plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(trial_dir, f'force_{c}.png'), dpi = 200)
        plt.close()

    if env.P.calculate_occlusion() > 0:
        count = 0
        while env.P.calculate_occlusion()>0:
            env.P.apply_force(env.force + count*5, env.location)
            count+=1
            
        if (max(abs(env.P.max_von_mises)) > 90*10**6):
            successful_stops += 1
        else:
            failed_stops += 1
            

    obs = env.reset()

print(f'Breaks: {breaks}')
print(f'Success: {success}')
print(f'Successful Stops: {successful_stops}')
print(f'Failed Stops: {failed_stops}')
