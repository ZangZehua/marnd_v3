import torch.nn as nn


class RND(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RND, self).__init__()
        # local novelty target net
        # using for set an anchor for calculating the novelty of local observation
        # params frozen, don't update
        self.local_novelty_target = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        for params in self.local_novelty_target.parameters():
            params.requires_grad = False

        # local novelty predict net
        # using for calculating the novelty of local observation $r_i^{local} = f_i^local(o_i^t)$
        # this subnet backwards 2 loss(from global loss and local loss)
        self.local_novelty_predict = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        target_novelty = self.local_novelty_target(x)
        predict_novelty = self.local_novelty_predict(x)

        rnd_loss = nn.functional.mse_loss(predict_novelty, target_novelty)
        return rnd_loss

    def get_rnd_reward(self, x):
        target_novelty = self.local_novelty_target(x)
        predict_novelty = self.local_novelty_predict(x)

        rnd_reward = nn.functional.mse_loss(predict_novelty, target_novelty, reduction='none').mean(dim=0, keepdim=True)
        return rnd_reward
