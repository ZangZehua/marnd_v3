import torch
import torch.nn as nn
import torch.nn.functional as F


class ImmacAgent(nn.Module):
    """
    input shape: [batch_size * n_agents, input_dim]
    output shape: [batch_size, n_agents, n_actions]
    hidden state shape: [batch_size, n_agents, hidden_dim]
    """

    def __init__(self, input_dim, scheme, args):
        super().__init__()
        self.args = args

        # curiosity generator, target network, 2 layers, frozen
        self.tar_fc1 = nn.Linear(input_dim, args.cur_hidden_dim)
        self.tar_fc2 = nn.Linear(args.cur_hidden_dim, args.cur_output_dim)

        for p in self.parameters():
            p.requires_grad = False

        self.obs_mean, self.obs_std = torch.zeros(scheme["obs"]["vshape"]).cuda(), torch.ones(scheme["obs"]["vshape"]).cuda()
        self.obs_count = 1e-4
        self.ext_count = 1e-4
        self.int_count = 1e-4
        self.padding_dim = input_dim - scheme["obs"]["vshape"]
        self.int_mean, self.int_std = torch.zeros(1).cuda(), torch.ones(1).cuda()
        self.ext_mean, self.ext_std = torch.zeros(1).cuda(), torch.ones(1).cuda()


        # curiosity generator, predictor network, 3 layers
        self.pred_fc1 = nn.Linear(input_dim, args.cur_hidden_dim)
        self.pred_dropout1 = nn.Dropout(p=0.5)
        self.pred_fc2 = nn.Linear(args.cur_hidden_dim, args.cur_hidden_dim)
        self.pred_dropout2 = nn.Dropout(p=0.5)
        self.pred_fc3 = nn.Linear(args.cur_hidden_dim, args.cur_output_dim)

        # external importance
        #self.ext_imp_fc1 = nn.Linear(input_dim, args.ext_imp_hidden_dim)
        self.ext_imp_fc1 = nn.Linear(input_dim, args.ext_imp_hidden_dim)
        self.ext_imp_fc2 = nn.Linear(args.ext_imp_hidden_dim, 1)

        # local Q
        self.q_fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.q_rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.q_fc2 = nn.Linear(args.rnn_hidden_dim * 2, args.n_actions)

    def forward(self, x, hidden):
        """
        x: [batch_size * n_agents, input_dim]
        hidden state: [batch_size, n_agents, hidden_dim]
        """
        x = F.relu(self.q_fc1(x))
        x = x.view(-1, x.size(-1))
        h_in = hidden.view(-1, self.args.rnn_hidden_dim)
        h_out = self.q_rnn(x, h_in)
        h_out = h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        return h_out

    def generate_message(self, x, hidden):
        """
        x: [batch_size * n_agents, input_dim]
        hidden: [batch_size, n_agents, rnn_hidden_dim]
        curiosity: [batch_size, n_agents]
        message: [batch_size, n_agents, rnn_hidden_dim]
        comm_rate: [batch_size, 1]
        pred_loss: int
        """
        obs_padding_mean = torch.zeros(self.padding_dim).cuda()
        obs_padding_std = torch.ones(self.padding_dim).cuda()
        obs_mean = torch.cat([self.obs_mean, obs_padding_mean])
        obs_std = torch.cat([self.obs_std, obs_padding_std])
        x_normalized = (x - obs_mean) / obs_std
        print(x_normalized.size())

        tar_val = F.relu(self.tar_fc1(x_normalized))
        tar_val = self.tar_fc2(tar_val)

        pred_val = F.relu(self.pred_fc1(x_normalized))
        pred_val = F.relu(self.pred_fc2(pred_val))
        pred_val = self.pred_fc3(pred_val)

        # pred loss
        pred_loss = F.mse_loss(pred_val, tar_val)

        # mean or sum
        curiosity = F.mse_loss(tar_val, pred_val, reduction='none').mean(dim=-1,
                                                                         keepdim=True)  # [batch_size * n_agents, 1]
        curiosity = curiosity.view(-1, self.args.n_agents, 1)  # [batch_size, n_agents, 1]

        curiosity_ = curiosity.expand_as(hidden)  # [batch_size, n_agents, rnn_hidden_dim]
        message = torch.where(curiosity_ > self.args.threshold, hidden, torch.zeros_like(hidden))

        curiosity = curiosity.squeeze(dim=-1)  # [batch_size, n_agents]
        comm_rate = torch.where(curiosity > self.args.threshold, torch.ones_like(curiosity),
                                torch.zeros_like(curiosity))
        comm_rate = comm_rate.mean(dim=-1, keepdim=True)  # [batch_size, 1]
        curiosity = torch.where(curiosity > self.args.threshold, curiosity,
                                torch.full_like(curiosity, -9999999))  # [batch_size, n_agents]
        return curiosity, message, comm_rate, pred_loss

    def generate_message_2(self, x, hidden):
        """
        importance = external importance + intrinsic importance(curiosity)
        x: [batch_size * n_agents, input_dim]
        hidden: [batch_size, n_agents, rnn_hidden_dim]
        importance: [batch_size, n_agents]
        message: [batch_size, n_agents, rnn_hidden_dim]
        comm_rate: [batch_size, 1]
        pred_loss: int
        """
        #observation normalization
        if self.args.observation_normalization:
            obs_padding_mean = torch.zeros(self.padding_dim).cuda()
            obs_padding_std = torch.ones(self.padding_dim).cuda()
            obs_mean = torch.cat([self.obs_mean, obs_padding_mean])
            obs_std = torch.cat([self.obs_std, obs_padding_std])
            x_normalized = (x - obs_mean) / obs_std

            tar_val = F.relu(self.tar_fc1(x_normalized))
            tar_val = self.tar_fc2(tar_val)

            pred_val = F.relu(self.pred_dropout1(self.pred_fc1(x_normalized)))
            pred_val = F.relu(self.pred_dropout2(self.pred_fc2(pred_val)))
            pred_val = self.pred_fc3(pred_val)
        else:
            tar_val = F.relu(self.tar_fc1(x))
            tar_val = self.tar_fc2(tar_val)

            pred_val = F.relu(self.pred_dropout1(self.pred_fc1(x)))
            pred_val = F.relu(self.pred_dropout2(self.pred_fc2(pred_val)))
            pred_val = self.pred_fc3(pred_val)

        # pred loss
        pred_loss = F.mse_loss(pred_val, tar_val)

        # external importance, using sigmoid activation to ensure ext_imp > 0
        ext_imp = F.relu(self.ext_imp_fc1(x))
        ext_imp = F.sigmoid(self.ext_imp_fc2(ext_imp))  #
        ext_imp = ext_imp.view(-1, self.args.n_agents, 1)

        #normalize the extrinsic value
        #ext_imp = ext_imp / self.ext_std

        int_imp = F.mse_loss(tar_val, pred_val, reduction='none').mean(dim=-1,
                                                                       keepdim=True)  # [batch_size * n_agents, 1]
        int_imp = int_imp.view(-1, self.args.n_agents, 1)  # [batch_size, n_agents, 1]
        #normalize the intrinsic value
        if self.args.val_normalization:
            int_imp = int_imp / self.int_std

        # now the two values' scales are continuous, combine
        imp = self.args.ext_lambd * ext_imp + self.args.int_lambd * int_imp.detach() # detach

        # message
        imp_ = imp.expand_as(hidden)  # [batch_size, n_agents, rnn_hidden_dim]
        message = torch.where(imp_ > self.args.threshold, hidden, torch.zeros_like(hidden))

        # comm_rate
        imp = imp.squeeze(dim=-1)  # [batch_size, n_agents]
        comm_rate = torch.where(imp > self.args.threshold, torch.ones_like(imp), torch.zeros_like(imp))
        comm_rate = comm_rate.mean(dim=-1, keepdim=True)  # [batch_size, 1]

        # imp = - flaot('inf') when agent don't send message
        imp = torch.where(imp > self.args.threshold, imp, torch.full_like(imp, -9999999))  # [batch_size, n_agents]
        return imp, message, comm_rate, pred_loss, ext_imp

    def combine(self, importance, message):
        """
        importance: [batch_size, n_agents]
        message: [batch_size, n_agents, rnn_hidden_dim]
        integrated_message: [batch_size, n_agents, rnn_hidden_dim]
        """
        imp = F.softmax(importance, dim=-1).view(-1, self.args.n_agents, 1)
        integ_msg = torch.mul(message, imp).sum(dim=1, keepdim=True)  # [batch_size, 1, rnn_hidden_dim]
        integ_msg = integ_msg.expand(-1, self.args.n_agents, -1)  # [batch_size, n_agents, rnn_hidden_dim]
        return integ_msg

    def local_q(self, hidden, integ_msg):
        """
        local_q: [batch_size, n_agents, n_actions]
        """
        x = torch.cat([hidden, integ_msg], dim=-1)
        local_q = self.q_fc2(x)
        return local_q

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.q_fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
