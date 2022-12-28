import torch.multiprocessing as mp
from actor_critic import ActorCritic
from ICM import ICM
from optimizer import SharedAdam
from ICM.memory import Memory
from isaacgym_sim.isaacgym_env import IsaacGymPlantEnv
import torch
import numpy as np
import time
import matplotlib.pyplot as plt

class ParallelEnv:
    def __init__(self, input_shape, n_actions, icm, simulation, n_envs=8):
        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        optimizer = SharedAdam(global_actor_critic.parameters())

        if not icm:
            global_icm = None
            icm_optimizer = None
        else:
            global_icm = ICM(*input_shape, n_actions, intrinsic_gain=1/4000)
            global_icm.share_memory()
            icm_optimizer = SharedAdam(global_icm.parameters())

        T_MAX = 20
        envs = []
        env_local_agents = []
        env_memories = []
        env_ep_done = []
        env_hx = []
        scores = []*n_envs
        ep_counters = []
        env_obs = [None]*n_envs
        local_icms = []

        for i in range(0, n_envs):
            envs.append(IsaacGymPlantEnv(simulation, i, observation_mode='Grayscale Image', action_mode='All Joints'))
            env_local_agents.append(ActorCritic(input_shape, n_actions))

            if icm:
                local_icms.append(ICM(input_shape[0], n_actions, intrinsic_gain=1/4000))
            # else:
            #     intrinsic_rewards.append(torch.zeros(1))
            #     algo = 'A3C'

            env_memories.append(Memory())

        max_eps, episode = 1000, 0
        # Go through an episode for all environments
        # Some may finish quicker than others
        env_t_steps = [0]*n_envs

        L_Fs = []
        L_Is = []
        ex_rew = []
        in_rew = []

        while episode < max_eps:
            env_ep_done = [False]*n_envs
            env_hx = [torch.zeros(1, 256)]*n_envs
            env_ep_steps = [0]*n_envs
            env_values = [None]*n_envs
            env_logprob = [None]*n_envs  
            actions = [None]*n_envs          

            for env_index in range(0, n_envs):
                # Keep the rest the same as per ICM m(40, 40)ultithread implementation
                env_obs[env_index] = envs[env_index].reset()

            # While each episode is not done (each timestep)
            while sum(env_ep_done) != n_envs:
                # Preliminary step to get actions assigned and started
                for env_index in range(0, n_envs):
                    obs = env_obs[env_index]
                    hx = env_hx[env_index]
                    state = torch.tensor([obs], dtype=torch.float)
                    local_agent = env_local_agents[env_index]
                    action, value, log_prob, hx = local_agent(state, hx)
                    env = envs[env_index]
                    env._take_action(action)
                    env_hx[env_index]=hx
                    env_values[env_index] = value
                    env_logprob[env_index] = log_prob
                    actions[env_index] = action

                t_0 = time.time()
                target=np.array(envs[0].S.target_angles)
                timeout = 0
                while timeout<10:
                    envs[0].S.sim_step(skip_images = False, step_printout=False)
                    timeout+=1
                end = np.array(envs[0].S.current_angles)
                t_f = time.time()

                for env_index in range(0, n_envs):
                    if not env_ep_done[env_index]:
                        memory = env_memories[env_index]
                        env = envs[env_index]
                        obs = env_obs[env_index]
                        log_prob = env_logprob[env_index]
                        value = env_values[env_index]
                        action = actions[env_index]

                        obs_, reward, done, info = env.step(action)

                        env_ep_done[env_index] = done
                        env_t_steps[env_index] += 1
                        env_ep_steps[env_index] += 1
                        # score += reward
                        # reward = 0  # turn off extrinsic rewards
                        memory.remember(obs, action, reward, obs_, value, log_prob)
                        env_obs[env_index] = obs_

                        with torch.autograd.set_detect_anomaly(True):
                            if env_ep_steps[env_index] % T_MAX == 0 or done:
                                local_agent = env_local_agents[env_index]

                                states, actions, rewards, new_states, values, log_probs = \
                                        memory.sample_memory()

                                if icm:
                                    local_icm = local_icms[env_index]

                                    intrinsic_rewards, L_I, L_F = \
                                            local_icm.calc_loss(states, new_states, actions)

                                    L_Fs.append(float(L_F))
                                    L_Is.append(float(L_I))

                                    plt.plot(L_Fs, label = 'Forward Loss')
                                    plt.plot(L_Is, label = 'Inverse Loss')
                                    plt.legend()
                                    plt.tight_layout()
                                    plt.savefig('out/losses.png')
                                    plt.close()
                                else:
                                    intrinsic_rewards = torch.zeros(len(rewards))

                                loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                                            log_probs, intrinsic_rewards+torch.tensor(rewards, dtype=torch.float32))
                                
                                in_rew=in_rew+list(intrinsic_rewards.detach().numpy())
                                plt.plot(in_rew, label = 'Intrinsic Reward')
                                
                                ex_rew=ex_rew+list(rewards)
                                
                                plt.plot(ex_rew, label = 'Extrinsic Reward')
                                plt.legend()
                                plt.tight_layout()
                                plt.savefig('out/rewards.png')
                                plt.close()
                                
                                optimizer.zero_grad()
                                hx = hx.detach_()
                                if icm:
                                    icm_optimizer.zero_grad()
                                    (L_I + L_F).backward()

                                loss.backward(retain_graph = True)
                                torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

                                loss.detach()

                                for local_param, global_param in zip(
                                                        local_agent.parameters(),
                                                        global_actor_critic.parameters()):
                                    global_param._grad = local_param.grad
                                optimizer.step()
                                local_agent.load_state_dict(global_actor_critic.state_dict())

                                if icm:
                                    for local_param, global_param in zip(
                                                            local_icm.parameters(),
                                                            global_icm.parameters()):
                                        global_param._grad = local_param.grad
                                    icm_optimizer.step()
                                    local_icm.load_state_dict(global_icm.state_dict())
                                memory.clear_memory()

            episode += 1
            
        