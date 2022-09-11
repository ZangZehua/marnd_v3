import copy
import torch
from torch.optim import RMSprop

from components.episode_buffer import EpisodeBatch
from components.rnd_model import RND
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.agent.q_fc1.parameters()) + list(mac.agent.q_rnn.parameters()) + list(
            mac.agent.q_fc2.parameters())  # local_q

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        global_rnd_input_dim = scheme["state"]["vshape"]
        self.global_rnd = RND(global_rnd_input_dim, args.rnd_hidden_dim, args.rnd_output_dim).cuda()

        # opt for local_q, mixer
        self.optimiser = RMSprop(params=filter(lambda x: x.requires_grad, self.params),
                                 lr=args.lr,
                                 alpha=args.optim_alpha,
                                 eps=args.optim_eps)

        self.local_rnd_params = list(mac.agent.local_rnd.parameters())
        # opt for local_rnd
        self.local_rnd_optimiser = RMSprop(params=filter(lambda x: x.requires_grad, self.local_rnd_params),
                                           lr=args.lr,
                                           alpha=args.optim_alpha,
                                           eps=args.optim_eps)
        self.global_rnd_params = list(self.global_rnd.parameters())
        # opt for global_rnd
        self.global_rnd_optimiser = RMSprop(params=filter(lambda x: x.requires_grad, self.global_rnd_params),
                                            lr=args.lr,
                                            alpha=args.optim_alpha,
                                            eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        # train ngu
        self.mac.train_ngu(batch)

        # Calculate estimated Q-Values(local_q) and local novelty
        mac_out = []
        local_novelty_loss = []
        local_novelty_reward = []
        ngu_reward = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs_t, local_novelty_loss_t, local_novelty_reward_t, ngu_reward_t \
                = self.mac.forward(batch, t=t, train_mode=True)
            mac_out.append(agent_outs_t)
            if not t == 0:
                local_novelty_loss.append(local_novelty_loss_t)
                local_novelty_reward.append(local_novelty_reward_t)
                ngu_reward.append(ngu_reward_t)
        mac_out = torch.stack(mac_out, dim=1)  # Concat over time
        local_novelty_loss = torch.stack(local_novelty_loss).mean()
        local_novelty_reward = torch.stack(local_novelty_reward).transpose(1, 2).transpose(0, 1)
        ngu_reward = torch.Tensor(ngu_reward).transpose(1, 2).transpose(0, 1)
        print("rnd reward", local_novelty_reward.shape)
        print("ngu reward", ngu_reward.shape)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = torch.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = torch.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = torch.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate the global novelty
        states = batch["state"][:, :-1]  # [batch_size, t, state_shape]
        states = states.transpose(0, 1)
        init_state = torch.zeros_like(states[0]).unsqueeze(0)
        states = torch.cat((init_state, states), dim=0)
        global_novelty_loss = []
        global_novelty_reward = []
        for t in range(1, batch.max_seq_length):
            global_novelty_loss_t = self.global_rnd(states[t])
            global_novelty_reward_t = self.global_rnd.get_rnd_reward(states[t])
            global_novelty_loss.append(global_novelty_loss_t)
            global_novelty_reward.append(global_novelty_reward_t)
        global_novelty_loss = torch.stack(global_novelty_loss).mean()
        global_novelty_reward = torch.stack(global_novelty_reward).transpose(1, 2).transpose(0, 1)

        # Calculate 1-step Q-Learning targets
        rewards += self.args.local_rnd_reward_lambda * local_novelty_reward + self.args.global_rnd_reward_lambda * global_novelty_reward
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.local_rnd_optimiser.zero_grad()
        local_novelty_loss.backward()
        self.local_rnd_optimiser.step()

        self.global_rnd_optimiser.zero_grad()
        global_novelty_loss.backward()
        self.global_rnd_optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.log_stats_t = t_env

    def _update_obs_mean_std(self, obs, mask):
        # obs
        # obs = obs[mask] # !important

        # batch_mean, batch_std, batch_count = np.mean(obs, axis=0), np.std(obs, axis=0), obs.shape[0]
        if self.args.observation_mask:
            obs = obs[:, :-1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
            mask = (mask.view(-1) > 0)
            # mask = torch.as_tensor(mask.view(-1), dtype=torch.bool)
            obs = obs[mask].reshape(-1, self.mac.agent.obs_mean.size(0))

            batch_mean, batch_std = torch.mean(obs, dim=0), torch.std(obs, dim=0)
            batch_count = obs.shape[0]
        else:
            batch_mean, batch_std = torch.mean(obs, dim=0), torch.std(obs, dim=0)
            batch_mean, batch_std = torch.mean(batch_mean, dim=0), torch.std(batch_std, dim=0)
            batch_mean, batch_std = torch.mean(batch_mean, dim=0), torch.std(batch_std, dim=0)
            batch_count = obs.shape[0] * obs.shape[1] * obs.shape[2]

        batch_var = batch_std ** 2
        var = self.mac.agent.obs_std ** 2

        delta = batch_mean - self.mac.agent.obs_mean
        tot_count = self.mac.agent.obs_count + batch_count

        new_mean = self.mac.agent.obs_mean + delta * batch_count / tot_count
        m_a = var * (self.mac.agent.obs_count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + delta ** 2 * self.mac.agent.obs_count * batch_count / (self.mac.agent.obs_count + batch_count)
        new_var = M2 / (self.mac.agent.obs_count + batch_count)

        new_count = batch_count + self.mac.agent.obs_count

        self.mac.agent.obs_count = new_count
        self.mac.agent.obs_mean = new_mean
        self.mac.agent.obs_std = torch.sqrt(new_var)

    def _update_val_std(self, ext_val):
        # print(ext_val.size(), int_val.size()) [46,32,3,1] [46]
        ext_batch_mean, ext_batch_std = torch.mean(ext_val), torch.std(ext_val)
        ext_batch_count = len(ext_val)
        ext_batch_var = ext_batch_std ** 2
        ext_var = self.mac.agent.ext_std ** 2

        ext_delta = ext_batch_mean - self.mac.agent.ext_mean
        ext_total_count = self.mac.agent.ext_count + ext_batch_count

        ext_new_mean = self.mac.agent.ext_mean + ext_delta * ext_batch_count / ext_total_count
        m_a = ext_var * (self.mac.agent.ext_count)
        m_b = ext_batch_var * (ext_batch_count)
        M2 = m_a + m_b + ext_delta ** 2 * self.mac.agent.ext_count * ext_batch_count / (
                self.mac.agent.ext_count + ext_batch_count)
        ext_new_var = M2 / (self.mac.agent.ext_count + ext_batch_count)

        self.mac.agent.ext_mean = ext_new_mean
        self.mac.agent.ext_std = torch.sqrt(ext_new_var)
        self.mac.agent.ext_count = ext_total_count

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, patorch):
        self.mac.save_models(patorch)
        if self.mixer is not None:
            torch.save(self.mixer.state_dict(), "{}/mixer.torch".format(patorch))
        torch.save(self.optimiser.state_dict(), "{}/opt.torch".format(patorch))

    def load_models(self, patorch):
        self.mac.load_models(patorch)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(patorch)
        if self.mixer is not None:
            self.mixer.load_state_dict(
                torch.load("{}/mixer.torch".format(patorch), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(
            torch.load("{}/opt.torch".format(patorch), map_location=lambda storage, loc: storage))
