import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

for file, icm in zip(['sawyer-precision/ep_log_env_0.csv', 'scslabubuntu2/ep_log_env_0.csv'], ['ICM', 'No_ICM']):
    ep_log = pd.read_csv(f'out/simulation/{file}')
    rewards = ep_log['Reward']
    windows = rewards.rolling(30)
    reward_moving = windows.mean().tolist()
    
    plt.plot(rewards, alpha = 0.3, color = 'gray', label = 'Rewards')
    plt.plot(reward_moving, alpha = 1, color = 'red', label = 'Moving average of rewards')
    plt.ylim((-5, 15))
    plt.xlim((0, 450))
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title(icm)
    plt.tight_layout()
    plt.savefig(f'out/{icm}.png')
    plt.close()

