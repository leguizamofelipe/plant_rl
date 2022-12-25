import gym
import numpy as np
import torch as T
from actor_critic import ActorCritic
from ICM import ICM
from ICM.memory import Memory
from isaacgym_sim.isaacgym_env import IsaacGymPlantEnv

def worker(env_n, input_shape, n_actions, global_agent, global_icm, optimizer, icm_optimizer, simulation, icm=False):
    
    T_MAX = 20
    
    env = IsaacGymPlantEnv(simulation)

    local_agent = ActorCritic(input_shape, n_actions)

    if icm:
        local_icm = ICM(input_shape[0], n_actions)
        algo = 'ICM'
    else:
        intrinsic_reward = T.zeros(1)
        algo = 'A3C'

    memory = Memory()

    t_steps, max_eps, episode, scores, avg_score = 0, 1000, 0, [], 0

    while episode < max_eps:
        obs = env.reset(env_n)
        hx = T.zeros(1, 256)
        score, done, ep_steps = 0, False, 0
        while not done:
            state = T.tensor([obs], dtype=T.float)
            action, value, log_prob, hx = local_agent(state, hx)
            obs_, reward, done, info = env.step(action, env_n)
            t_steps += 1
            ep_steps += 1
            score += reward
            reward = 0  # turn off extrinsic rewards
            memory.remember(obs, action, reward, obs_, value, log_prob)
            obs = obs_
            if ep_steps % T_MAX == 0 or done:
                states, actions, rewards, new_states, values, log_probs = \
                        memory.sample_memory()
                if icm:
                    intrinsic_reward, L_I, L_F = \
                            local_icm.calc_loss(states, new_states, actions)

                loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                             log_probs, intrinsic_reward)

                optimizer.zero_grad()
                hx = hx.detach_()
                if icm:
                    icm_optimizer.zero_grad()
                    (L_I + L_F).backward()

                loss.backward()
                T.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                for local_param, global_param in zip(
                                        local_agent.parameters(),
                                        global_agent.parameters()):
                    global_param._grad = local_param.grad
                optimizer.step()
                local_agent.load_state_dict(global_agent.state_dict())

                if icm:
                    for local_param, global_param in zip(
                                            local_icm.parameters(),
                                            global_icm.parameters()):
                        global_param._grad = local_param.grad
                    icm_optimizer.step()
                    local_icm.load_state_dict(global_icm.state_dict())
                memory.clear_memory()

        episode += 1
    
    
    print('yeet')