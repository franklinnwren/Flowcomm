import numpy as np
from torch import Tensor
from torch.autograd import Variable
import time

class ReplayBuffer(object):
    """
    Replay Buffer for multi-agent RL with parallel rollouts
    """
    def __init__(self, max_steps, num_agents, obs_dims, ac_dims, full_obs_dims, messg_dim):
        """
        Inputs:
            max_steps (int): Maximum number of timepoints to store in buffer
            num_agents (int): Number of agents in environment
            obs_dims (list of ints): number of obervation dimensions for each
                                     agent
            ac_dims (list of ints): number of action dimensions for each agent
        """
        self.max_steps      = max_steps
        self.num_agents     = num_agents
        self.obs_buffs      = []
        self.ac_buffs       = []
        self.rew_buffs      = []
        self.next_obs_buffs = []
        self.done_buffs     = []

        self.obs_messg_buffs           = []       
        self.next_obs_messg_buffs      = []  
     
        for odim, adim in zip(obs_dims, ac_dims):
            self.obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.ac_buffs.append(np.zeros((max_steps, adim), dtype=np.float32))
            self.rew_buffs.append(np.zeros(max_steps, dtype=np.float32))
            self.next_obs_buffs.append(np.zeros((max_steps, odim), dtype=np.float32))
            self.done_buffs.append(np.zeros(max_steps, dtype=np.uint8))

            self.obs_messg_buffs.append(np.zeros((max_steps, odim+messg_dim), dtype=np.float32))
            self.next_obs_messg_buffs.append(np.zeros((max_steps, odim+messg_dim), dtype=np.float32))

        self.full_obs_buffs           = np.zeros((max_steps, full_obs_dims), dtype=np.float32)    
        self.matrix_A_buffs           = np.zeros((max_steps, self.num_agents, self.num_agents), dtype=np.float32)    
        self.log_matrix_A_probs_buffs = np.zeros((max_steps, 1), dtype=np.float32) 
   
        self.filled_i = 0  # index of first empty location in buffer (last index when full)
        self.curr_i = 0  # current index to write to (ovewrite oldest data)

    def __len__(self):
        return self.filled_i

    def push(self, observations, actions, rewards, next_observations, dones, matrix_A, log_matrix_A_probs, full_observations, observations_message, next_observations_message):
        nentries = observations.shape[0]  # handle multiple parallel environments
        if self.curr_i + nentries > self.max_steps:
            rollover = self.max_steps - self.curr_i # num of indices to roll over
            for agent_i in range(self.num_agents):

                self.obs_buffs[agent_i]             = np.roll(self.obs_buffs[agent_i],rollover, axis=0)
                self.ac_buffs[agent_i]              = np.roll(self.ac_buffs[agent_i],rollover, axis=0)
                self.rew_buffs[agent_i]             = np.roll(self.rew_buffs[agent_i],rollover)
                self.next_obs_buffs[agent_i]        = np.roll(self.next_obs_buffs[agent_i], rollover, axis=0)
                self.done_buffs[agent_i]            = np.roll(self.done_buffs[agent_i],rollover)

                self.obs_messg_buffs[agent_i]       = np.roll(self.obs_messg_buffs[agent_i], rollover, axis=0)
                self.next_obs_messg_buffs[agent_i]  = np.roll(self.next_obs_messg_buffs[agent_i], rollover, axis=0)

            self.full_obs_buffs             = np.roll(self.full_obs_buffs, rollover, axis=0)
            self.matrix_A_buffs             = np.roll(self.matrix_A_buffs, rollover, axis=0)
            self.log_matrix_A_probs_buffs   = np.roll(self.log_matrix_A_probs_buffs, rollover)

            self.curr_i = 0
            self.filled_i = self.max_steps

        for agent_i in range(self.num_agents):
            self.obs_buffs[agent_i][self.curr_i:self.curr_i + nentries]              = np.vstack(observations[:, agent_i])
            # actions are already batched by agent, so they are indexed differently
            self.ac_buffs[agent_i][self.curr_i:self.curr_i + nentries]               = actions[agent_i]
            self.rew_buffs[agent_i][self.curr_i:self.curr_i + nentries]              = rewards[:, agent_i]
            self.next_obs_buffs[agent_i][self.curr_i:self.curr_i + nentries]         = np.vstack(next_observations[:, agent_i])
            self.done_buffs[agent_i][self.curr_i:self.curr_i + nentries]             = dones[:, agent_i]

            self.obs_messg_buffs[agent_i][self.curr_i:self.curr_i + nentries]        = np.vstack(observations_message[:, agent_i])
            self.next_obs_messg_buffs[agent_i][self.curr_i:self.curr_i + nentries]   = np.vstack(next_observations_message[:, agent_i])
        
        self.full_obs_buffs[self.curr_i:self.curr_i + nentries]                      = full_observations[:]                
        self.matrix_A_buffs[self.curr_i:self.curr_i + nentries]                      = matrix_A[:]
        self.log_matrix_A_probs_buffs[self.curr_i:self.curr_i + nentries]            = log_matrix_A_probs[:]

        self.curr_i += nentries
        if self.filled_i < self.max_steps:
            self.filled_i += nentries
        if self.curr_i == self.max_steps:
            self.curr_i = 0


    def sample(self, N, to_gpu=False, norm_rews=True):
        inds = np.random.choice(np.arange(self.filled_i), size=N,
                                replace=True)
        if to_gpu:
            cast = lambda x: Variable(Tensor(x), requires_grad=False).cuda()
        else:
            cast = lambda x: Variable(Tensor(x), requires_grad=False)
        if norm_rews:
            ret_rews = [cast((self.rew_buffs[i][inds] -
                              self.rew_buffs[i][:self.filled_i].mean()) /
                             self.rew_buffs[i][:self.filled_i].std())
                        for i in range(self.num_agents)]
        else:
            ret_rews = [cast(self.rew_buffs[i][inds]) for i in range(self.num_agents)]
        return ([cast(self.obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.ac_buffs[i][inds]) for i in range(self.num_agents)],
                ret_rews,
                [cast(self.next_obs_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.done_buffs[i][inds]) for i in range(self.num_agents)],

                cast(self.matrix_A_buffs[inds]),
                cast(self.log_matrix_A_probs_buffs[inds]),
                cast(self.full_obs_buffs[inds]),

                [cast(self.obs_messg_buffs[i][inds]) for i in range(self.num_agents)],
                [cast(self.next_obs_messg_buffs[i][inds]) for i in range(self.num_agents)],
               )

    def get_average_rewards(self, N):
        if self.filled_i == self.max_steps:
            inds = np.arange(self.curr_i - N, self.curr_i)  # allow for negative indexing
        else:
            inds = np.arange(max(0, self.curr_i - N), self.curr_i)
        return [self.rew_buffs[i][inds].mean() for i in range(self.num_agents)]
