"""
This code is modified base on
https://github.com/shariqiqbal2810/MAAC, written by Shariq Iqbal
Which is introduced and explained in the paper: https://arxiv.org/abs/1810.02912
"""
import argparse
import torch
import os
import numpy as np
from gym.spaces import Box, Discrete
from pathlib import Path
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from utils.make_env import make_env
from utils.buffer import ReplayBuffer
from utils.env_wrappers import SubprocVecEnv, DummyVecEnv
from algorithms.attention_sac import AttentionSAC
import time
from graph_model import GraphFlows


def make_parallel_env(env_id, n_rollout_threads, seed):
    def get_env_fn(rank):
        def init_env():
            env = make_env(env_id, discrete_action=True)
            env.seed(seed + rank * 1000)
            np.random.seed(seed + rank * 1000)
            return env
        return init_env
    if n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(n_rollout_threads)])


def test_message_pooling(hidden_state, agent_num, config):

    hidden_state_pooling         = []

    for i in range(len(hidden_state)):
        hidden_state_array_pooling = []
        hidden_state_i = hidden_state[i].reshape(len(hidden_state[i][0]))
        for j in range(len(hidden_state_i)):
            if j % config.pooling_interval !=0:
                continue
            else:
                pooling_state = hidden_state_i[j] - hidden_state_i[j]
                for index in range(config.pooling_interval):
                    pooling_state = pooling_state + hidden_state_i[j+index]
                                
                hidden_state_array_pooling.append(pooling_state/config.pooling_interval)
        hidden_state_pooling.append(np.array(hidden_state_array_pooling,dtype='float32').reshape(1,len(hidden_state_array_pooling)))

    for i in range(agent_num):
        if i == 0:
            hidden_state_vstack = hidden_state_pooling[i]
        else:
            hidden_state_vstack = np.vstack((hidden_state_vstack, hidden_state_pooling[i]))

    return hidden_state_vstack


def train_message_pooling(hidden_state, agent_num, config):

    hidden_state_pooling         = np.zeros((agent_num, config.n_rollout_threads, config.critic_hidden_dim // config.pooling_interval))

    for i in range(len(hidden_state)):
        hidden_state_array_pooling = np.zeros((config.n_rollout_threads, config.critic_hidden_dim // config.pooling_interval))
        hidden_state_i = hidden_state[i].reshape((config.n_rollout_threads,len(hidden_state[i][0])))

        digit = 0
        for j in range(hidden_state_i.shape[1]):
            if j % config.pooling_interval !=0:
                continue
            else:
                for thread in range(config.n_rollout_threads):

                    pooling_state = np.zeros(1)
                    for index in range(config.pooling_interval):
                        pooling_state = pooling_state + hidden_state_i[thread][j+index]
                                
                    hidden_state_array_pooling[thread][digit]= pooling_state / config.pooling_interval
                digit += 1
        hidden_state_pooling[i] = hidden_state_array_pooling

    hidden_state_vstack = np.zeros((config.n_rollout_threads, agent_num, config.critic_hidden_dim // config.pooling_interval))
    for thread in range(config.n_rollout_threads):
        for i in range(agent_num):
            hidden_state_vstack[thread][i] = hidden_state_pooling[i][thread]

    return hidden_state_vstack


def run(config):
    model_dir = Path('./models') / config.env_id / config.model_name
    if not model_dir.exists():
        run_num = 1
    else:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in
                         model_dir.iterdir() if
                         str(folder.name).startswith('run')]
        if len(exst_run_nums) == 0:
            run_num = 1
        else:
            run_num = max(exst_run_nums) + 1
    curr_run = 'run%i' % run_num
    run_dir = model_dir / curr_run
    log_dir = run_dir / 'logs'
    os.makedirs(log_dir)
    logger = SummaryWriter(str(log_dir))

    torch.manual_seed(run_num)
    np.random.seed(run_num)
    env        = make_parallel_env(config.env_id, config.n_rollout_threads, run_num)

    test_env   = make_parallel_env(config.env_id, 1, run_num)

    full_obs_dim = 0
    for obsp in env.observation_space:
        full_obs_dim = full_obs_dim + obsp.shape[0]
    agent_ob_list = [obsp.shape[0] for obsp in env.observation_space]

    model = AttentionSAC.init_from_env(env,
                                       tau=config.tau,
                                       pi_lr=config.pi_lr,
                                       q_lr=config.q_lr,
                                       gamma=config.gamma,
                                       pol_hidden_dim=config.pol_hidden_dim,
                                       critic_hidden_dim=config.critic_hidden_dim,
                                       attend_heads=config.attend_heads,
                                       reward_scale=config.reward_scale, messg_length = config.critic_hidden_dim // config.pooling_interval)

    GraphFlow_model = GraphFlows(n_s=full_obs_dim, n_agent=len(agent_ob_list), n_step=config.batch_size)

    replay_buffer   = ReplayBuffer(config.buffer_length, model.nagents,
                                 [obsp.shape[0] for obsp in env.observation_space],
                                 [acsp.shape[0] if isinstance(acsp, Box) else acsp.n
                                  for acsp in env.action_space], full_obs_dim, config.critic_hidden_dim // config.pooling_interval)

    t = 0
    for ep_i in range(0, config.n_episodes, config.n_rollout_threads):
        print("Episodes %i-%i of %i" % (ep_i + 1,
                                        ep_i + 1 + config.n_rollout_threads,
                                        config.n_episodes))

        if (ep_i) % config.test_interval == 0:
            for test_ep_i in range(10):
                obs = test_env.reset()
                model.prep_rollouts(device='cpu')
                agent_0_comm = 0
                agent_1_comm = 0
                agent_2_comm = 0
                agent_3_comm = 0

                torch_hidden_state             = torch.zeros((model.nagents,config.critic_hidden_dim // config.pooling_interval))

                for test_et_i in range(config.episode_length):

                    obs_messg_list             = []

                    for j in range(obs.shape[0]):

                        for i in range(model.nagents):
                            if i == 0:
                                full_obs        = obs[j][i]
                            else:
                                full_obs        = np.hstack((full_obs, obs[j][i]))

                        matrix_A, log_matrix_A_probs = GraphFlow_model.forward([full_obs])

                        if j == 0:
                            matrix_A_numpy = matrix_A.detach().numpy()
                            for row in range(matrix_A_numpy.shape[0]):
                                for col in range(matrix_A_numpy.shape[1]):
                                    if col == 0:
                                        if matrix_A_numpy[row][col] == 1:
                                            agent_0_comm += 1
                                    if col == 1:
                                        if matrix_A_numpy[row][col] == 1:
                                            agent_1_comm += 1
                                    if col == 2:
                                        if matrix_A_numpy[row][col] == 1:
                                            agent_2_comm += 1
                                    if col == 3:
                                        if matrix_A_numpy[row][col] == 1:
                                            agent_3_comm += 1

                            if test_et_i == config.episode_length-1:
                                print(agent_0_comm, agent_1_comm, agent_2_comm, agent_3_comm)

                        messg_split                  = torch.matmul(matrix_A, torch_hidden_state).squeeze()

                        obs_messg                    = []

                        for i in range(model.nagents):
                           messg_i                   =  messg_split[i]
                           obs_messg.append(np.concatenate([obs[j][i], messg_i.detach().numpy()]))

                        obs_messg_list.append(obs_messg)

                    obs_messg_list           = np.array(obs_messg_list)#,dtype=object)
                    torch_obs = [Variable(torch.Tensor(np.vstack(obs_messg_list[:, i])),
                                          requires_grad=False)
                                 for i in range(model.nagents)]

                    # get actions as torch Variables
                    torch_agent_actions = model.step(torch_obs, explore=True)

                    hidden_state                 = [ac[1] for ac in torch_agent_actions]
                   
                    hidden_state_vstack          = test_message_pooling(hidden_state, model.nagents, config)

                    torch_hidden_state           = torch.tensor(hidden_state_vstack, dtype=torch.float32, device= 'cpu')
                    # convert actions to numpy arrays
                    agent_actions = [ac[0].data.numpy() for ac in torch_agent_actions]
            
                    # rearrange actions to be per environment
                    actions = [[ac[i] for ac in agent_actions] for i in range(1)]
                    next_obs, rewards, dones, infos = test_env.step(actions)
                    obs = next_obs

            test_env.close()

        obs = env.reset()
        model.prep_rollouts(device='cpu')

        agent_0_comm = 0
        agent_1_comm = 0
        agent_2_comm = 0
        agent_3_comm = 0

        torch_hidden_state             = torch.zeros((config.n_rollout_threads, model.nagents, config.critic_hidden_dim // config.pooling_interval))
        next_obs_messg_list            = None
        for et_i in range(config.episode_length):
            # rearrange observations to be per agent, and convert to torch Variable

            matrix_A_list              = [] 
            log_matrix_A_probs_list    = []
            full_obs_list              = []
            obs_messg_list             = []
            
            for j in range(obs.shape[0]):

                for i in range(model.nagents):
                    if i == 0:
                        full_obs        = obs[j][i]
                    else:
                        full_obs        = np.hstack((full_obs, obs[j][i]))

                full_obs_list.append(full_obs)
                matrix_A, log_matrix_A_probs = GraphFlow_model.forward([full_obs])

                if j == -1:
                    matrix_A_numpy = matrix_A.detach().numpy()
                    for row in range(matrix_A_numpy.shape[0]):
                        for col in range(matrix_A_numpy.shape[1]):
                            if col == 0:
                                if matrix_A_numpy[row][col] == 1:
                                    agent_0_comm += 1
                            if col == 1:
                                if matrix_A_numpy[row][col] == 1:
                                    agent_1_comm += 1
                            if col == 2:
                                if matrix_A_numpy[row][col] == 1:
                                    agent_2_comm += 1
                            if col == 3:
                                if matrix_A_numpy[row][col] == 1:
                                    agent_3_comm += 1

                    if et_i == config.episode_length-1:
                        print(agent_0_comm, agent_1_comm, agent_2_comm, agent_3_comm)

                matrix_A_list.append(matrix_A.detach().numpy())
                log_matrix_A_probs_list.append(log_matrix_A_probs.detach().numpy())

                messg_split                  = torch.matmul(matrix_A, torch_hidden_state[j]).squeeze()

                obs_messg                    = []

                for i in range(model.nagents):
                   messg_i                   =  messg_split[i]
                   obs_messg.append(np.concatenate([obs[j][i], messg_i.detach().numpy()]))

                obs_messg_list.append(obs_messg)

            matrix_A_list            = np.array(matrix_A_list)
            log_matrix_A_probs_list  = np.array(log_matrix_A_probs_list)
            obs_messg_list           = np.array(obs_messg_list)#,dtype=object)
            full_obs_list            = np.array(full_obs_list)
           
            if next_obs_messg_list is not None:
                obs_messg_list = next_obs_messg_list

            torch_obs = [Variable(torch.Tensor(np.vstack(obs_messg_list[:, i])),
                                  requires_grad=False)
                         for i in range(model.nagents)]

            # get actions as torch Variables
            torch_agent_actions = model.step(torch_obs, explore=True)
            # convert actions to numpy arrays
            hidden_state                 = [ac[1] for ac in torch_agent_actions]

            hidden_state_vstack          = train_message_pooling(hidden_state, model.nagents, config)

            torch_hidden_state           = torch.tensor(hidden_state_vstack, dtype=torch.float32, device= 'cpu')

            agent_actions = [ac[0].data.numpy() for ac in torch_agent_actions]

            # rearrange actions to be per environment
            actions = [[ac[i] for ac in agent_actions] for i in range(config.n_rollout_threads)]
            next_obs, rewards, dones, infos = env.step(actions)

            next_obs_messg_list                     = []

            next_torch_hidden_state                 = torch_hidden_state

            for j in range(next_obs.shape[0]):

                for i in range(model.nagents):
                    if i == 0:
                        next_full_obs        = next_obs[j][i]
                    else:
                        next_full_obs        = np.hstack((next_full_obs, next_obs[j][i]))

                next_matrix_A, next_log_matrix_A_probs = GraphFlow_model.forward([next_full_obs])

                next_messg_split                        = torch.matmul(matrix_A, next_torch_hidden_state[j]).squeeze()

                next_obs_messg  = []
                for i in range(model.nagents):
                    next_messg_i                       =  next_messg_split[i]
                    next_obs_messg.append(np.concatenate([next_obs[j][i], next_messg_i.detach().numpy()]))

                next_obs_messg_list.append(next_obs_messg)

            next_obs_messg_list  = np.array(next_obs_messg_list)#,dtype=object)

            replay_buffer.push(obs, agent_actions, rewards, next_obs, dones, matrix_A_list, log_matrix_A_probs_list, full_obs_list, obs_messg_list, next_obs_messg_list)
            obs = next_obs

            t += config.n_rollout_threads
            if (len(replay_buffer) >= config.batch_size and
                (t % config.steps_per_update) < config.n_rollout_threads):
                if config.use_gpu:
                    model.prep_training(device='gpu')
                else:
                    model.prep_training(device='cpu')
                for u_i in range(config.num_updates):
                    sample = replay_buffer.sample(config.batch_size,
                                                  to_gpu=config.use_gpu)
                    model.update_critic(sample, logger=logger)

                    matrix_As_sample, log_matrix_A_probs_sample, Q_mean, full_obs_sample = model.update_policies(sample, logger=logger)
                    graph_loss = GraphFlow_model.backward(obs=full_obs_sample, qs=Q_mean.detach().numpy(), As=matrix_As_sample, log_As_probs=log_matrix_A_probs_sample)
                    model.update_all_targets()
                logger.add_scalar('Graph_Loss/graph_loss', graph_loss, ep_i)
                model.prep_rollouts(device='cpu')
        
        ep_rews = replay_buffer.get_average_rewards(
            config.episode_length * config.n_rollout_threads)
        print(ep_rews)
        for a_i, a_ep_rew in enumerate(ep_rews):
            logger.add_scalar('agent%i/mean_episode_rewards' % a_i,
                              a_ep_rew * config.episode_length, ep_i)

        if ep_i % config.save_interval < config.n_rollout_threads:
            model.prep_rollouts(device='cpu')
            os.makedirs(run_dir / 'incremental', exist_ok=True)
            model.save(run_dir / 'incremental' / ('model_ep%i.pt' % (ep_i + 1)))
            model.save(run_dir / 'model.pt')
            GraphFlow_model.save(run_dir / 'graph.pt')

    model.save(run_dir / 'model.pt')
    env.close()
    logger.export_scalars_to_json(str(log_dir / 'summary.json'))
    logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("env_id", help="Name of environment")
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    parser.add_argument("--n_rollout_threads", default=12, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=15625, type=int)
    parser.add_argument("--episode_length", default=64, type=int)
    parser.add_argument("--steps_per_update", default=4, type=int)
    parser.add_argument("--num_updates", default=4, type=int,
                        help="Number of updates per update cycle")
    parser.add_argument("--batch_size",
                        default=1024, type=int,
                        help="Batch size for training")
    parser.add_argument("--save_interval", default=1000, type=int)
    parser.add_argument("--pol_hidden_dim", default=64, type=int)
    parser.add_argument("--critic_hidden_dim", default=64, type=int)
    parser.add_argument("--attend_heads", default=4, type=int)
    parser.add_argument("--pi_lr", default=0.001, type=float)
    parser.add_argument("--q_lr", default=0.001, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--reward_scale", default=100., type=float)
    parser.add_argument("--use_gpu", action='store_true')
    parser.add_argument("--test_interval", default=120, type=int)
    parser.add_argument("--pooling_interval", default=1, type=int)
    config = parser.parse_args()

    run(config)
