from isaacgym_sim.isaacgym_env import IsaacGymPlantEnv
from stable_baselines3 import PPO
from stable_baselines3.ppo.policies import MlpPolicy

env = IsaacGymPlantEnv()

time_steps = 1000000

model = PPO(MlpPolicy, env, verbose = 1, device = 'cuda')

model.learn(total_timesteps=int(time_steps), n_eval_episodes = 30)