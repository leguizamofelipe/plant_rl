import torch.multiprocessing as mp
from actor_critic import ActorCritic
from ICM import ICM
from optimizer import SharedAdam
from ICM.memory import Memory
from isaacgym_sim.isaacgym_env import IsaacGymPlantEnv
import torch

class ParallelEnv:
    def __init__(self, input_shape, n_actions, icm, simulation, n_envs=8):
        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        optimizer = SharedAdam(global_actor_critic.parameters())

        if not icm:
            global_icm = None
            icm_optimizer = None
        else:
            global_icm = ICM(*input_shape, n_actions)
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
            envs.append(IsaacGymPlantEnv(simulation, i))
            env_local_agents.append(ActorCritic(input_shape, n_actions))

            if icm:
                local_icms.append(ICM(input_shape[0], n_actions))
                algo = 'ICM'
            else:
                intrinsic_rewards.append(torch.zeros(1))
                algo = 'A3C'

            env_memories.append(Memory())

            env_ep_done.append(False)
            env_hx.append(torch.zeros(1, 256))

        max_eps, episode = 1000, 0
        # Go through an episode for all environments
        # Some may finish quicker than others
        env_t_steps = [0]*n_envs

        while episode < max_eps:
            for env_index in range(0, n_envs):
                # Keep the rest the same as per ICM multithread implementation
                env_obs[env_index] = envs[env_index].reset()
                env_hx[env_index] = torch.zeros(1, 256)
                
            # scores = [0]*n_envs
            env_ep_done = [False]*n_envs
            env_ep_steps = [0]*n_envs            

            while sum(env_ep_done) != n_envs:
                for env_index in range(0, n_envs):
                    if not env_ep_done[env_index]:
                        local_agent = env_local_agents[env_index]
                        memory = env_memories[env_index]
                        env = envs[env_index]
                        obs = env_obs[env_index]
                        hx = env_hx[env_index]

                        state = torch.tensor([obs], dtype=torch.float)
                        action, value, log_prob, hx = local_agent(state, hx)
                        obs_, reward, done, info = env.step(action)
                        env_ep_done[env_index] = done
                        env_t_steps[env_index] += 1
                        env_ep_steps[env_index] += 1
                        # score += reward
                        reward = 0  # turn off extrinsic rewards
                        memory.remember(obs, action, reward, obs_, value, log_prob)
                        env_obs[env_index] = obs_
                        if env_ep_steps[env_index] % T_MAX == 0 or done:
                            local_icm = local_icms[env_index]
                            states, actions, rewards, new_states, values, log_probs = \
                                    memory.sample_memory()
                            if icm:
                                intrinsic_rewards, L_I, L_F = \
                                        local_icm.calc_loss(states, new_states, actions)

                            loss = local_agent.calc_loss(obs, hx, done, rewards, values,
                                                        log_probs, intrinsic_rewards)

                            optimizer.zero_grad()
                            hx = hx.detach_()
                            if icm:
                                icm_optimizer.zero_grad()
                                (L_I + L_F).backward()

                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(local_agent.parameters(), 40)

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
            
        