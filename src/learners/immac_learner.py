import copy
from src.components.episode_buffer import EpisodeBatch
from src.modules.mixers.vdn import VDNMixer
from src.modules.mixers.qmix import QMixer
import torch as th
from torch.optim import RMSprop


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
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

        self.optimiser = RMSprop(params=filter(lambda x: x.requires_grad, self.params),
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

        if self.args.observation_normalization:
            self._update_obs_mean_std(batch['obs'], mask)
        # Calculate estimated Q-Values
        mac_out = []
        comm_rate = []
        pred_loss = []
        ext_imp = []
        importance = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agents_out_t, comm_rate_t, pred_loss_t, ext_imp_t, importance_t = self.mac.forward(batch, t=t, train_mode=True)
            mac_out.append(agents_out_t)
            comm_rate.append(comm_rate_t)
            pred_loss.append(pred_loss_t)
            ext_imp.append(ext_imp_t)
            importance.append(importance_t)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        #update the std of intrinsic and extrinsic value


        comm_rate = th.stack(comm_rate, dim=1) # [batch_size, max_seq_length, 1]
        pred_loss = th.stack(pred_loss) # [max_seq_length]
        ext_imp = th.stack(ext_imp)
        importance = th.stack(importance)

        if self.args.val_normalization:
            self._update_val_std(pred_loss)

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        # q learning loss + prediction loss

        pred_loss = pred_loss.mean()
        q_loss = (masked_td_error ** 2).sum() / mask.sum()
        loss = q_loss + pred_loss

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            # self.logger.log_stat("loss_entropy", loss_entropy.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item() / mask_elems), t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_action_qvals * mask).sum().item() / (mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item() / (mask_elems * self.args.n_agents),
                                 t_env)
            self.logger.log_stat("comm_rate", comm_rate.mean().item(), t_env)
            self.logger.log_stat("pred_loss", pred_loss.item(), t_env)
            self.logger.log_stat("ext_imp", ext_imp.mean().item(), t_env)
            self.logger.log_stat("importance", importance.mean().item(), t_env)

            self.log_stats_t = t_env

    def _update_obs_mean_std(self, obs, mask):
        # obs
        # obs = obs[mask] # !important

        # batch_mean, batch_std, batch_count = np.mean(obs, axis=0), np.std(obs, axis=0), obs.shape[0]
        if self.args.observation_mask:
            obs = obs[:, :-1]
            obs = obs.reshape(obs.shape[0] * obs.shape[1], -1)
            mask = (mask.view(-1) > 0)
            # mask = th.as_tensor(mask.view(-1), dtype=th.bool)
            obs = obs[mask].reshape(-1, self.mac.agent.obs_mean.size(0))

            batch_mean, batch_std = th.mean(obs, dim=0), th.std(obs, dim=0)
            batch_count = obs.shape[0]
        else:
            batch_mean, batch_std = th.mean(obs, dim=0), th.std(obs, dim=0)
            batch_mean, batch_std = th.mean(batch_mean, dim=0), th.std(batch_std, dim=0)
            batch_mean, batch_std = th.mean(batch_mean, dim=0), th.std(batch_std, dim=0)
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
        self.mac.agent.obs_std = th.sqrt(new_var)

    def _update_val_std(self, ext_val):
        #print(ext_val.size(), int_val.size()) [46,32,3,1] [46]
        ext_batch_mean, ext_batch_std = th.mean(ext_val), th.std(ext_val)
        ext_batch_count = len(ext_val)
        ext_batch_var = ext_batch_std ** 2
        ext_var = self.mac.agent.ext_std ** 2

        ext_delta = ext_batch_mean - self.mac.agent.ext_mean
        ext_total_count = self.mac.agent.ext_count + ext_batch_count

        ext_new_mean = self.mac.agent.ext_mean + ext_delta * ext_batch_count / ext_total_count
        m_a = ext_var * (self.mac.agent.ext_count)
        m_b = ext_batch_var * (ext_batch_count)
        M2 = m_a + m_b + ext_delta ** 2 * self.mac.agent.ext_count * ext_batch_count / (self.mac.agent.ext_count + ext_batch_count)
        ext_new_var = M2 / (self.mac.agent.ext_count + ext_batch_count)

        self.mac.agent.ext_mean = ext_new_mean
        self.mac.agent.ext_std = th.sqrt(ext_new_var)
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

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
