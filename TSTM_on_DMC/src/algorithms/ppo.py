import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import algorithms.modules as m
import utils


def atanh(x, eps=1e-6):
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


class RolloutBuffer(object):
    def __init__(self, obs_shape, action_shape, capacity):
        self.capacity = capacity
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.logp = np.zeros((capacity, 1), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.terminals = np.zeros((capacity, 1), dtype=np.float32)
        self.values = np.zeros((capacity, 1), dtype=np.float32)
        self.next_values = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0

    def reset(self):
        self.ptr = 0

    @property
    def full(self):
        return self.ptr >= self.capacity

    def add(self, obs, action, logp, reward, done, terminal, value, next_value):
        if self.full:
            return
        i = self.ptr
        self.obs[i] = np.array(obs, copy=False)
        self.actions[i] = action
        self.logp[i] = logp
        self.rewards[i] = reward
        self.dones[i] = done
        self.terminals[i] = terminal
        self.values[i] = value
        self.next_values[i] = next_value
        self.ptr += 1


class PPOActor(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.encoder = encoder
        self.action_dim = action_shape[0]
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.mu = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim),
        )

        self.log_std_param = nn.Parameter(torch.zeros(self.action_dim))

        self.mu.apply(m.weight_init)

    def _log_std(self):
        return torch.clamp(self.log_std_param, self.log_std_min, self.log_std_max)

    def forward(self, obs, detach=False):
        h = self.encoder(obs, detach=detach)
        mu = self.mu(h)
        log_std = self._log_std().expand_as(mu)
        return mu, log_std

    def dist(self, obs, detach=False):
        mu, log_std = self.forward(obs, detach=detach)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std), mu, log_std


class PPOCritic(nn.Module):
    def __init__(self, encoder, hidden_dim):
        super().__init__()
        self.encoder = encoder
        self.v = nn.Sequential(
            nn.Linear(self.encoder.out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.v.apply(m.weight_init)

    def forward(self, obs, detach=False):
        h = self.encoder(obs, detach=detach)
        v = self.v(h).squeeze(-1)
        return v


class PPOActorCritic(nn.Module):
    def __init__(self, actor_encoder, critic_encoder, action_shape, hidden_dim, log_std_min, log_std_max):
        super().__init__()
        self.actor = PPOActor(
            encoder=actor_encoder,
            action_shape=action_shape,
            hidden_dim=hidden_dim,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
        )
        self.critic = PPOCritic(
            encoder=critic_encoder,
            hidden_dim=hidden_dim,
        )

    def forward(self, obs, detach_actor=False, detach_critic=False):
        mu, log_std = self.actor(obs, detach=detach_actor)
        v = self.critic(obs, detach=detach_critic)
        return mu, log_std, v

    def dist(self, obs, detach_actor=False):
        return self.actor.dist(obs, detach=detach_actor)

    def sample(self, obs):
        dist, mu, log_std = self.dist(obs, detach_actor=False)
        noise = torch.randn_like(mu)
        u = mu + noise * log_std.exp()
        a = torch.tanh(u)
        logp_u = dist.log_prob(u).sum(dim=-1)
        logp = logp_u - torch.log(1.0 - a.pow(2) + 1e-6).sum(dim=-1)
        v = self.critic(obs, detach=False)
        return a, logp, v, mu, log_std

    def evaluate_actions(self, obs, actions, detach_actor=False, detach_critic=False):
        dist, mu, log_std = self.dist(obs, detach_actor=detach_actor)
        u = atanh(actions)
        logp_u = dist.log_prob(u).sum(dim=-1)
        logp = logp_u - torch.log(1.0 - actions.pow(2) + 1e-6).sum(dim=-1)
        ent = dist.entropy().sum(dim=-1)
        v = self.critic(obs, detach=detach_critic)
        return logp, ent, v, mu, log_std


class PPO(object):
    def __init__(self, obs_shape, action_shape, args):
        self.device = torch.device("cuda")

        self.discount = args.discount
        self.gae_lambda = getattr(args, "gae_lambda", 0.95)
        self.clip_eps = getattr(args, "clip_eps", 0.2)
        self.ppo_epochs = getattr(args, "ppo_epochs", 10)
        self.ppo_minibatch_size = getattr(args, "ppo_minibatch_size", 64)
        self.value_coef = getattr(args, "value_coef", 0.5)
        self.entropy_coef = getattr(args, "entropy_coef", 0.0)
        self.max_grad_norm = getattr(args, "max_grad_norm", 0.5)
        self.use_clipped_value_loss = getattr(args, "ppo_use_clipped_value_loss", True)

        self.detach_actor_trunk = bool(getattr(args, "ppo_detach_actor_trunk", False))

        shared_cnn = m.SharedCNN(obs_shape, args.num_shared_layers, args.num_filters).cuda()
        head_cnn = m.HeadCNN(shared_cnn.out_shape, args.num_head_layers, args.num_filters).cuda()
        actor_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )
        critic_encoder = m.Encoder(
            shared_cnn,
            head_cnn,
            m.RLProjection(head_cnn.out_shape, args.projection_dim),
        )

        self.actor_critic = PPOActorCritic(
            actor_encoder=actor_encoder,
            critic_encoder=critic_encoder,
            action_shape=action_shape,
            hidden_dim=args.hidden_dim,
            log_std_min=args.actor_log_std_min,
            log_std_max=args.actor_log_std_max,
        ).cuda()

        self.actor = self.actor_critic.actor
        self.critic = self.actor_critic.critic

        self._optim_params = []
        _seen = set()
        for p in self.actor_critic.parameters():
            pid = id(p)
            if pid in _seen:
                continue
            self._optim_params.append(p)
            _seen.add(pid)

        self.optimizer = torch.optim.Adam(self._optim_params, lr=getattr(args, "ppo_lr", 3e-4))

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor_critic.train(training)

    def eval(self):
        self.train(False)

    def _obs_to_input(self, obs):
        if isinstance(obs, utils.LazyFrames):
            _obs = np.array(obs)
        else:
            _obs = obs
        _obs = torch.as_tensor(_obs, device=self.device)
        if _obs.dtype == torch.uint8:
            _obs = _obs.float()
        else:
            _obs = _obs.float()
        _obs = _obs.unsqueeze(0)
        return _obs

    def act(self, obs):
        obs_t = self._obs_to_input(obs)
        with torch.no_grad():
            a, logp, v, _, _ = self.actor_critic.sample(obs_t)
        return a.cpu().numpy().flatten(), float(logp.item()), float(v.item())

    def get_value(self, obs):
        obs_t = self._obs_to_input(obs)
        with torch.no_grad():
            _, _, v = self.actor_critic.forward(obs_t)
        return float(v.item())

    def select_action(self, obs):
        obs_t = self._obs_to_input(obs)
        with torch.no_grad():
            mu, _, _ = self.actor_critic.forward(obs_t)
            a = torch.tanh(mu)
        return a.cpu().numpy().flatten()

    def sample_action(self, obs):
        a, _, _ = self.act(obs)
        return a

    def _compute_gae(self, rewards, dones, terminals, values, next_values):
        T = rewards.shape[0]
        adv = np.zeros((T, 1), dtype=np.float32)
        last_gae = 0.0
        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - dones[t]
            bootstrap = 1.0 - terminals[t]
            delta = rewards[t] + self.discount * next_values[t] * bootstrap - values[t]
            last_gae = delta + self.discount * self.gae_lambda * nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def update(self, rollout_buffer, L, step):
        if not getattr(rollout_buffer, "full", False):
            return

        rewards = rollout_buffer.rewards
        dones = rollout_buffer.dones
        terminals = rollout_buffer.terminals
        values = rollout_buffer.values
        next_values = rollout_buffer.next_values
        adv, ret = self._compute_gae(rewards, dones, terminals, values, next_values)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        obs = torch.as_tensor(rollout_buffer.obs, device=self.device).float()
        actions = torch.as_tensor(rollout_buffer.actions, device=self.device).float()
        old_logp = torch.as_tensor(rollout_buffer.logp, device=self.device).float().squeeze(-1)
        old_v = torch.as_tensor(rollout_buffer.values, device=self.device).float().squeeze(-1)
        adv_t = torch.as_tensor(adv, device=self.device).float().squeeze(-1)
        ret_t = torch.as_tensor(ret, device=self.device).float().squeeze(-1)

        n = obs.shape[0]
        idx = np.arange(n)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.ppo_minibatch_size):
                end = start + self.ppo_minibatch_size
                mb = idx[start:end]

                o = obs[mb]
                a = actions[mb]
                old_lp = old_logp[mb]
                old_v_mb = old_v[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]

                new_lp, ent, v_pred, _, log_std = self.actor_critic.evaluate_actions(
                    o,
                    a,
                    detach_actor=self.detach_actor_trunk,
                    detach_critic=False,
                )
                ratio = torch.exp(new_lp - old_lp)

                s1 = ratio * adv_mb
                s2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_mb
                policy_loss = -torch.min(s1, s2).mean()

                if self.use_clipped_value_loss:
                    v_pred_clipped = old_v_mb + (v_pred - old_v_mb).clamp(-self.clip_eps, self.clip_eps)
                    value_losses = (v_pred - ret_mb).pow(2)
                    value_losses_clipped = (v_pred_clipped - ret_mb).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (ret_mb - v_pred).pow(2).mean()

                entropy_loss = -ent.mean()

                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self._optim_params, self.max_grad_norm)
                self.optimizer.step()

                if L is not None:
                    L.log("train_actor/loss", policy_loss, step)
                    L.log("train_critic/loss", value_loss, step)
                    L.log("train_actor/entropy", ent.mean(), step)
                    with torch.no_grad():
                        approx_kl = (old_lp - new_lp).mean()
                        clipfrac = (torch.abs(ratio - 1.0) > self.clip_eps).float().mean()
                        L.log("train_ppo/approx_kl", approx_kl, step)
                        L.log("train_ppo/clipfrac", clipfrac, step)
                        L.log("train_ppo/ratio_mean", ratio.mean(), step)
                        L.log("train_ppo/ratio_std", ratio.std(unbiased=False), step)
                        L.log("train_ppo/adv_mean", adv_mb.mean(), step)
                        L.log("train_ppo/adv_std", adv_mb.std(unbiased=False), step)
                        L.log("train_ppo/v_pred_mean", v_pred.mean(), step)
                        L.log("train_ppo/ret_mean", ret_mb.mean(), step)
                        L.log("train_ppo/log_std_mean", log_std.mean(), step)
                        L.log("train_ppo/log_std_min", log_std.min(), step)
                        L.log("train_ppo/log_std_max", log_std.max(), step)
                        L.log("train_ppo/grad_norm", grad_norm, step)

        rollout_buffer.reset()
