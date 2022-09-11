from typing import List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim, Tensor


class NGUEmbeddingModel(nn.Module):
    def __init__(self, obs_size, hidden_dim, num_outputs):
        """
        NUG model
        :param obs_size: the obs vector of original obs/state
        :param num_outputs: the action number
        """
        super(NGUEmbeddingModel, self).__init__()
        self.obs_size = obs_size
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(obs_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.last = nn.Linear(hidden_dim * 2, num_outputs)

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x1, x2):
        """
        using for training the NGU model
        :param x1: obs/state
        :param x2: next_obs/next_state
        :return: predicted action
        """
        x1 = self.embedding(x1)
        x2 = self.embedding(x2)
        x = torch.cat([x1, x2], dim=2)
        x = self.last(x)
        return nn.Softmax(dim=2)(x)

    def embedding(self, x):
        """
        using for getting the embedding feature
        :param x: obs/state
        :return: feature: [batch_size, 32]
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

    def train_model(self, batch):
        """
        using for training the NGU model
        :param batch: memory batch, need obs/state, next_obs/next_state, action
        :return: loss, using for logging
        """
        batch_size = torch.stack(batch.state).size()[0]
        # last 5 in sequence
        states = torch.stack(batch.state).view(batch_size, config.sequence_length, self.obs_size)[:, -5:, :]
        next_states = torch.stack(batch.next_state).view(batch_size, config.sequence_length, self.obs_size)[:, -5:, :]
        actions = torch.stack(batch.action).view(batch_size, config.sequence_length, -1).long()[:, -5:, :]

        self.optimizer.zero_grad()
        net_out = self.forward(states, next_states)
        actions_one_hot = torch.squeeze(F.one_hot(actions, self.num_outputs)).float()
        loss = nn.MSELoss()(net_out, actions_one_hot)
        loss.backward()
        self.optimizer.step()
        return loss.item()


def compute_intrinsic_reward(
        episodic_memory: List,
        current_c_state: Tensor,
        k=10,
        kernel_cluster_distance=0.008,
        kernel_epsilon=0.0001,
        c=0.001,
        sm=8,
) -> float:
    ngu_reward = []  # len(agents_num)
    ep_memory = torch.stack(episodic_memory).transpose(0, 1)  # [n_agents, batch_size, obs_dim]
    for agent in range(ep_memory.shape[0]):
        memory = ep_memory[0]
        #         print("memory", memory, memory.shape)
        obs = current_c_state[0]
        #         print("obs", obs, obs.shape)

        state_dist = [(c_state, torch.dist(c_state, obs)) for c_state in memory]
        state_dist.sort(key=lambda x: x[1])
        state_dist = state_dist[:k]
        dist = [d[1].item() for d in state_dist]
        dist = np.array(dist)

        dist = dist / np.mean(dist)

        dist = np.max(dist - kernel_cluster_distance, 0)
        kernel = kernel_epsilon / (dist + kernel_epsilon)
        s = np.sqrt(np.sum(kernel)) + c

        if np.isnan(s) or s > sm:
            s = 0
        else:
            s = 1 / s
        ngu_reward.append(s)
    return ngu_reward
