import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules as m
from augmentations import random_overlay
from torch.distributions import Normal
from tstm_segment.temporal_segmentation_network import SimpleCNN_ConvLSTM
from algorithms.ppo import PPO


def compute_INV_loss(z_a, z_b, lambda_val=1.25, mu_val=1.25, nu_val=0.005, gamma_val=1.0, eps=1e-4):
    sim_loss = F.mse_loss(z_a, z_b)

    std_z_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_z_b = torch.sqrt(z_b.var(dim=0) + eps)
    std_loss_a = torch.mean(F.relu(gamma_val - std_z_a))
    std_loss_b = torch.mean(F.relu(gamma_val - std_z_b))
    std_loss = std_loss_a + std_loss_b

    N, D = z_a.shape
    z_a_centered = z_a - z_a.mean(dim=0)
    z_b_centered = z_b - z_b.mean(dim=0)
    cov_z_a = (z_a_centered.T @ z_a_centered) / (N - 1)
    cov_z_b = (z_b_centered.T @ z_b_centered) / (N - 1)
    cov_loss_a = cov_z_a.fill_diagonal_(0).pow_(2).sum() / D
    cov_loss_b = cov_z_b.fill_diagonal_(0).pow_(2).sum() / D
    cov_loss = cov_loss_a + cov_loss_b

    total_loss = lambda_val * sim_loss + mu_val * std_loss + nu_val * cov_loss
    return total_loss, sim_loss, std_loss, cov_loss


class TSTM_PPO(PPO):
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)

        self.device = torch.device("cuda")
        self.policy_consistency_weight = getattr(args, "policy_consistency_weight", 1.0)
        self.inv_loss_weight = getattr(args, "INV_loss_weight", 0.1)
        self.inv_lambda = getattr(args, "INV_lambda", 1.25)
        self.inv_mu = getattr(args, "INV_mu", 1.25)
        self.inv_rho = getattr(args, "INV_rho", 0.005)
        self.inv_gamma = getattr(args, "INV_gamma", 1.0)

        self.temporal_mask_chunk_size = getattr(args, "temporal_mask_chunk_size", 64)
        self.temporal_mask_amp = bool(getattr(args, "temporal_mask_amp", True))

        encoder_output_dim = self.critic.encoder.out_dim
        projector_hidden_dim = getattr(args, "projector_hidden_dim", 2048)
        projector_output_dim = getattr(args, "projector_output_dim", 2048)
        self.projector = m.Projector(
            input_dim=encoder_output_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        ).to(self.device)

        self._optim_params = []
        _seen = set()
        for p in list(self.actor_critic.parameters()) + list(self.projector.parameters()):
            pid = id(p)
            if pid in _seen:
                continue
            self._optim_params.append(p)
            _seen.add(pid)

        self.optimizer = torch.optim.Adam(
            self._optim_params,
            lr=getattr(args, "ppo_lr", 3e-4),
        )

        self._load_temporal_model(args)

        p = float(getattr(args, "temporal_mask_threshold", 0.5))
        p = float(np.clip(p, 1e-6, 1.0 - 1e-6))
        self.temporal_mask_threshold = p
        self.temporal_mask_logit_threshold = float(np.log(p / (1.0 - p)))

    def _load_temporal_model(self, args):
        temporal_model_path = getattr(args, "temporal_model_path", None)
        if temporal_model_path is None or not os.path.exists(temporal_model_path):
            raise ValueError(f"temporal model path invalid or not provided: {temporal_model_path}")
        checkpoint = torch.load(temporal_model_path, map_location=self.device)
        state_dict = checkpoint["model_state_dict"]
        hidden_dim = checkpoint["hidden_dim"]

        self.temporal_model = SimpleCNN_ConvLSTM(
            input_channels=3,
            num_classes=1,
            hidden_dim=hidden_dim,
            kernel_size=3,
        ).to(self.device)

        self.temporal_model.load_state_dict(state_dict)
        self.temporal_model.freeze()
        self.temporal_model.eval()

    def apply_mask(self, obs, test_env=False):
        B, C_total, H, W = obs.shape
        num_frames = C_total // 3
        if C_total % 3 != 0:
            raise ValueError(f"expected C_total to be divisible by 3 (RGB frames), got C_total={C_total}")
        chunk_size = int(self.temporal_mask_chunk_size) if self.temporal_mask_chunk_size is not None else B
        if chunk_size <= 0:
            chunk_size = B

        obs_masked_chunks = []
        with torch.no_grad():
            for start in range(0, B, chunk_size):
                end = min(start + chunk_size, B)
                obs_chunk = obs[start:end].clone()
                b = obs_chunk.shape[0]
                obs_seq = obs_chunk.view(b, num_frames, 3, H, W)

                if obs_seq.dtype == torch.uint8 or obs_seq.max() > 1.0:
                    obs_seq_temporal = obs_seq.float() / 255.0
                else:
                    obs_seq_temporal = obs_seq if obs_seq.dtype == torch.float32 else obs_seq.float()

                amp_enabled = bool(self.temporal_mask_amp) and obs_seq_temporal.is_cuda
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    pred_masks, _ = self.temporal_model(obs_seq_temporal)
                masks = (pred_masks > float(self.temporal_mask_logit_threshold)).float()

                obs_seq_for_mask = obs_seq if obs_seq.dtype == torch.float32 else obs_seq.float()
                obs_masked = obs_seq_for_mask.mul_(masks)
                obs_masked = obs_masked.view(b, C_total, H, W)
                obs_masked_chunks.append(obs_masked)

        return torch.cat(obs_masked_chunks, dim=0)

    def act(self, obs):
        obs_t = self._obs_to_input(obs)
        obs_t = self.apply_mask(obs_t)
        with torch.no_grad():
            a, logp, v, _, _ = self.actor_critic.sample(obs_t)
        return a.cpu().numpy().flatten(), float(logp.item()), float(v.item())

    def get_value(self, obs):
        obs_t = self._obs_to_input(obs)
        obs_t = self.apply_mask(obs_t)
        with torch.no_grad():
            _, _, v = self.actor_critic.forward(obs_t)
        return float(v.item())

    def select_action(self, obs):
        obs_t = self._obs_to_input(obs)
        obs_t = self.apply_mask(obs_t, test_env=True)
        with torch.no_grad():
            mu, _, _ = self.actor_critic.forward(obs_t)
            a = torch.tanh(mu)
        return a.cpu().numpy().flatten()

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

        obs_raw = torch.as_tensor(rollout_buffer.obs, device=self.device).float()
        actions = torch.as_tensor(rollout_buffer.actions, device=self.device).float()
        old_logp = torch.as_tensor(rollout_buffer.logp, device=self.device).float().squeeze(-1)
        old_v = torch.as_tensor(rollout_buffer.values, device=self.device).float().squeeze(-1)
        adv_t = torch.as_tensor(adv, device=self.device).float().squeeze(-1)
        ret_t = torch.as_tensor(ret, device=self.device).float().squeeze(-1)

        with torch.no_grad():
            obs_aug = random_overlay(obs_raw.clone())
            obs_masked = self.apply_mask(obs_raw)
            obs_aug_masked = self.apply_mask(obs_aug)

        n = obs_masked.shape[0]
        idx = np.arange(n)

        for _ in range(self.ppo_epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.ppo_minibatch_size):
                end = start + self.ppo_minibatch_size
                mb = idx[start:end]

                o = obs_masked[mb]
                o_aug = obs_aug_masked[mb]
                a = actions[mb]
                old_lp = old_logp[mb]
                old_v_mb = old_v[mb]
                adv_mb = adv_t[mb]
                ret_mb = ret_t[mb]

                new_lp, ent, v_pred, mu, log_std = self.actor_critic.evaluate_actions(
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

                y_cat = self.critic.encoder(torch.cat([o, o_aug], dim=0), detach=False)
                z_cat = self.projector(y_cat)
                z_orig = z_cat[: o.shape[0]]
                z_aug = z_cat[o.shape[0] :]
                inv_total_loss, inv_sim_loss, inv_var_loss, inv_cov_loss = compute_INV_loss(
                    z_orig,
                    z_aug,
                    self.inv_lambda,
                    self.inv_mu,
                    self.inv_rho,
                    self.inv_gamma,
                )

                aug_mu, aug_log_std = self.actor_critic.actor(o_aug, detach=self.detach_actor_trunk)
                std = log_std.exp()
                std = torch.max(std, torch.ones_like(std) * 1e-3)
                target_dist = Normal(torch.tanh(mu.detach()), std.detach())
                aug_std = aug_log_std.exp()
                aug_std = torch.max(aug_std, torch.ones_like(aug_std) * 1e-3)
                current_dist = Normal(torch.tanh(aug_mu), aug_std)
                kl_loss = torch.distributions.kl_divergence(target_dist, current_dist).sum(dim=-1).mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                    + self.inv_loss_weight * inv_total_loss
                    + self.policy_consistency_weight * kl_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self._optim_params,
                    self.max_grad_norm,
                )
                self.optimizer.step()

                if L is not None:
                    L.log("train_actor/loss", policy_loss, step)
                    L.log("train_critic/loss", value_loss, step)
                    L.log("train_actor/entropy", ent.mean(), step)
                    L.log("train_inv/total_loss", inv_total_loss, step)
                    L.log("train_inv/invariance_loss", inv_sim_loss, step)
                    L.log("train_inv/variance_loss", inv_var_loss, step)
                    L.log("train_inv/covariance_loss", inv_cov_loss, step)
                    L.log("train/kl_loss", kl_loss, step)
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
