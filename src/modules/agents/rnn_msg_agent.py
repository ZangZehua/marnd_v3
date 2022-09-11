import torch
import torch.nn as nn
import torch.nn.functional as F


class RnnMsgAgent(nn.Module):
    """
    input shape: [batch_size, in_feature]
    output shape: [batch_size, n_actions]
    hidden state shape: [batch_size, hidden_dim]
    """

    def __init__(self, input_dim, args, scheme):
        super().__init__()

        self.args = args
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.fc_value = nn.Linear(args.rnn_hidden_dim, args.n_value)
        self.fc_key = nn.Linear(args.rnn_hidden_dim, args.n_key)
        self.fc_query = nn.Linear(args.rnn_hidden_dim, args.n_query)

        self.fc_attn = nn.Linear(args.n_query + args.n_key * args.n_agents, args.n_agents)

        self.fc_attn_combine = nn.Linear(args.n_value + args.rnn_hidden_dim, args.rnn_hidden_dim)

        # used when ablate 'shortcut' connection
        # self.fc_attn_combine = nn.Linear(args.n_value, args.rnn_hidden_dim)

        # curiosity generator, target network, 2 layers, frozen
        self.tar_fc1 = nn.Linear(input_dim, args.cur_hidden_dim)
        self.tar_fc2 = nn.Linear(args.cur_hidden_dim, args.cur_output_dim)
        # curiosity generator, predictor network, 3 layers
        self.pred_fc1 = nn.Linear(input_dim, args.cur_hidden_dim)
        self.pred_dropout1 = nn.Dropout(p=0.5)
        self.pred_fc2 = nn.Linear(args.cur_hidden_dim, args.cur_hidden_dim)
        self.pred_dropout2 = nn.Dropout(p=0.5)
        self.pred_fc3 = nn.Linear(args.cur_hidden_dim, args.cur_output_dim)

        self.obs_mean, self.obs_std = torch.zeros(scheme["obs"]["vshape"]).cuda(), torch.ones(
            scheme["obs"]["vshape"]).cuda()
        self.obs_count = 1e-4
        self.ext_count = 1e-4
        self.int_count = 1e-4
        self.padding_dim = input_dim - scheme["obs"]["vshape"]
        self.int_mean, self.int_std = torch.zeros(1).cuda(), torch.ones(1).cuda()
        self.ext_mean, self.ext_std = torch.zeros(1).cuda(), torch.ones(1).cuda()

    def forward(self, x, hidden):
        """
        hidden state: [batch_size, n_agents, hidden_dim]
        q_without_communication
        """
        x = F.relu(self.fc1(x))
        h_in = hidden.view(-1, self.args.rnn_hidden_dim)
        h_out = self.rnn(x, h_in)
        h_out = h_out.view(-1, self.args.n_agents, self.args.rnn_hidden_dim)
        return h_out

    def q_without_communication(self, h_out):
        q_without_comm = self.fc2(h_out)
        return q_without_comm

    def communicate(self, hidden):
        """
        input: hidden [batch_size, n_agents, hidden_dim]
        output: key, value, signature
        """
        key = self.fc_key(hidden)
        value = self.fc_value(hidden)
        query = self.fc_query(hidden)

        return key, value, query

    def aggregate(self, query, key, value, hidden, x):
        """
        query: [batch_size, n_agents, n_query]
        key: [batch_size, n_agents, n_key]
        value: [batch_size, n_agents, n_value]
        """
        #extrinsic attention weights (i.e.vffac weight)
        n_agents = self.args.n_agents
        _key = torch.cat([key[:, i, :] for i in range(n_agents)], dim=-1).unsqueeze(1).repeat(1, n_agents, 1)
        query_key = torch.cat([query, _key], dim=-1)  # [batch_size, n_agents, n_query + n_agents*n_key]

        # attention weights
        attn_weights = F.softmax(self.fc_attn(query_key), dim=-1)  # [batch_size, n_agents, n_agents]


        #intrinsic attention weights (i.e. immac attention weights)
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
        int_imp = F.mse_loss(tar_val, pred_val, reduction='none').mean(dim=-1,
                                                                       keepdim=True)  # [batch_size * n_agents, 1]
        int_imp = int_imp.view(-1, self.args.n_agents)    #[batch_size, n_agents]

        if self.args.val_normalization:
            int_imp = int_imp / self.int_std

        int_imp = F.softmax(int_imp, dim=-1).view(-1, self.args.n_agents, 1)
        int_imp = int_imp.repeat(1, 1, self.args.n_agents)    #[batch_size, n_agents, n_agents]
        int_imp = int_imp.permute(0, 2, 1)
        #int_imp = int_imp.view(-1, self.args.n_agents, self.args.n_agents)
        #softmax
        #combine extrinsic and intrinsic weights
        attn_combined = self.args.ext_lambd * attn_weights + self.args.int_lambd * int_imp.detach()    #where to detach

        # attentional value
        attn_applied = torch.bmm(attn_combined, value)  # [batch_size, n_agents, n_value]

        # shortcut connection: combine with agent's own hidden
        attn_combined = torch.cat([attn_applied, hidden], dim=-1)

        # used when ablate 'shortcut' connection
        # attn_combined = attn_applied

        attn_combined = F.relu(self.fc_attn_combine(attn_combined))

        # mlp, output Q
        q = self.fc2(attn_combined)  # [batch_size, n_agents, n_actions]
        return q

    def init_hidden(self):
        # trick, create hidden state on same device
        # batch size: 1
        return self.fc1.weight.new_zeros(1, self.args.rnn_hidden_dim)
