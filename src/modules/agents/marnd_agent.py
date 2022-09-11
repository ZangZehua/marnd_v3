import torch
import torch.nn as nn
import torch.nn.functional as F

from components.rnd_model import RND
from components.ngu_embedding_model import NGUEmbeddingModel, compute_intrinsic_reward


class MarndAgent(nn.Module):
    """
    input shape: [batch_size * n_agents, input_dim]
    output shape: [batch_size, n_agents, n_actions]
    hidden state shape: [batch_size, n_agents, hidden_dim]
    """

    def __init__(self, input_dim, scheme, args):
        super().__init__()
        self.args = args
        self.obs_mean, self.obs_std = torch.zeros(scheme["obs"]["vshape"]).cuda(), torch.ones(
            scheme["obs"]["vshape"]).cuda()
        self.obs_count = 1e-4
        self.ext_count = 1e-4
        self.int_count = 1e-4
        self.padding_dim = input_dim - scheme["obs"]["vshape"]
        self.int_mean, self.int_std = torch.zeros(1).cuda(), torch.ones(1).cuda()
        self.ext_mean, self.ext_std = torch.zeros(1).cuda(), torch.ones(1).cuda()

        # local Q net
        # using for calculating the local Q
        self.q_fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.q_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        # rnd model
        self.local_rnd = RND(input_dim, args.rnd_hidden_dim, args.rnd_output_dim)

        # ngu model
        self.ngu_memory = []
        self.ngu = NGUEmbeddingModel(input_dim, args.ngu_embedding_dim, args.n_actions)

    def forward(self, x, hidden):
        """

        :param x: [batch_size * n_agents, input_dim] [6, 192]
        :param hidden: [batch_size, n_agents, hidden_dim]
        :return: local_q: [batch_size, n_agents, n_actions]
                 h_out: [batch_size, n_agents, n_actions]
                 rnd_loss: [batch_size]
                 rnd_reward: [batch_size, n_agents]
        """
        local_q = F.relu(self.q_fc1(x))
        local_q = local_q.view(-1, local_q.size(-1))
        h_in = hidden.view(-1, self.args.rnn_hidden_dim)
        h_out = self.q_rnn(local_q, h_in)
        local_q = self.q_fc2(h_out).unsqueeze(0)

        rnd_loss = self.local_rnd(x)
        rnd_reward = self.local_rnd.get_rnd_reward(x)

        if x.shape[0] == self.args.n_agents:  # for train mode
            obs_ngu_embedding = self.ngu.embedding(x)

            self.ngu_memory.append(obs_ngu_embedding)
            ngu_reward = compute_intrinsic_reward(self.ngu_memory, obs_ngu_embedding)
        else:
            ngu_reward = [0 for i in range(self.args.n_agents)]  # for test mode
        return local_q, h_out, rnd_loss, rnd_reward, ngu_reward

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.q_fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)

    def clear_buffer(self):
        self.ngu_memory = []

    def train_ngu(self, batch):
        loss = self.ngu.train_model(batch)
        return loss


