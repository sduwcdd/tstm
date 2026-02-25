import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import algorithms.modules as m
from algorithms.sac import SAC
from augmentations import random_overlay
import numpy as np
from torch.distributions import MultivariateNormal
from tstm_segment.temporal_segmentation_network import SimpleCNN_ConvLSTM


def compute_INV_loss(z_a, z_b, lambda_val=1.25, mu_val=1.25, nu_val=0.005, gamma_val=1.0, eps=1e-4):
    """INV loss with invariance, variance (hinge on std), and covariance penalties."""
    # Invariance loss
    sim_loss = F.mse_loss(z_a, z_b)

    # Variance loss
    std_z_a = torch.sqrt(z_a.var(dim=0) + eps)
    std_z_b = torch.sqrt(z_b.var(dim=0) + eps)
    std_loss_a = torch.mean(F.relu(gamma_val - std_z_a))
    std_loss_b = torch.mean(F.relu(gamma_val - std_z_b))
    std_loss = std_loss_a + std_loss_b

    # Covariance loss
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


class TSTM(SAC):
    def __init__(self, obs_shape, action_shape, args):
        # init SAC
        SAC.__init__(self, obs_shape, action_shape, args)
        # IIEC/INV parameters
        self.device = torch.device("cuda")
        self.policy_consistency_weight = getattr(args, 'policy_consistency_weight', 1.0)
        self.inv_loss_weight = getattr(args, 'INV_loss_weight', 0.1)
        self.inv_lambda = getattr(args, 'INV_lambda', 1.25)
        self.inv_mu = getattr(args, 'INV_mu', 1.25)
        self.inv_rho = getattr(args, 'INV_rho', 0.005)
        self.inv_gamma = getattr(args, 'INV_gamma', 1.0)
        # Projection head parameters
        encoder_output_dim = self.critic.encoder.out_dim
        projector_hidden_dim = getattr(args, 'projector_hidden_dim', 2048)
        projector_output_dim = getattr(args, 'projector_output_dim', 2048)
        self.projector = m.Projector(
            input_dim=encoder_output_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim
        ).to(self.device)
        self.critic_optimizer.add_param_group({'params': self.projector.parameters()})

        # load temporal segmentation model
        self._load_temporal_model(args)

    def _load_temporal_model(self, args):
        """load temporal segmentation model"""
        temporal_model_path = getattr(args, 'temporal_model_path', None)
        if temporal_model_path is None or not os.path.exists(temporal_model_path):
            raise ValueError(f"temporal model path invalid or not provided: {temporal_model_path}")
        checkpoint = torch.load(temporal_model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        hidden_dim = checkpoint['hidden_dim']

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
        # Strictly require 5 frames (C=15). Raise error to avoid implicit padding/cropping.
        if num_frames != 5:
            raise ValueError(f"expected 5 frames (C=15), got {num_frames} frames (C={C_total})")
        with torch.no_grad():
            obs = obs.clone()
            obs_seq = obs.view(B, 5, 3, H, W)

            # Normalize to [0, 1] if input is uint8 or has values > 1.0
            if obs_seq.dtype == torch.uint8 or obs_seq.max() > 1.0:
                obs_seq_temporal = obs_seq.float() / 255.0
            else:
                obs_seq_temporal = obs_seq if obs_seq.dtype == torch.float32 else obs_seq.float()

            pred_masks, _ = self.temporal_model(obs_seq_temporal)
            masks = (pred_masks > 0.0).float()

            obs_seq_for_mask = obs_seq if obs_seq.dtype == torch.float32 else obs_seq.float()
            obs_masked = obs_seq_for_mask.mul_(masks)
            obs_masked = obs_masked.view(B, 15, H, W)  # reshape back to [B, 15, H, W]
        return obs_masked

    def update_critic(self, obs, obs_masked, obs_aug1, obs_aug1_masked, action, reward, next_obs_masked, not_done, L=None, step=None):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs_masked)
            target_Q1, target_Q2 = self.critic_target(next_obs_masked, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        obs_cat_masked = utils.cat(obs_masked, obs_aug1_masked)
        action = utils.cat(action, action)
        target_Q = utils.cat(target_Q, target_Q)
        current_Q1, current_Q2 = self.critic(obs_cat_masked, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        obs_inv_cat = utils.cat(obs_masked, obs_aug1_masked)
        y_cat = self.critic.encoder(obs_inv_cat)
        z_cat = self.projector(y_cat)
        batch_size = obs_masked.shape[0]
        z_orig = z_cat[:batch_size]
        z_aug = z_cat[batch_size:]
        inv_total_loss, inv_sim_loss, inv_var_loss, inv_cov_loss = compute_INV_loss(
            z_orig, z_aug, self.inv_lambda, self.inv_mu, self.inv_rho, self.inv_gamma
        )
        update_loss = critic_loss + self.inv_loss_weight * inv_total_loss
        
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
            L.log('train_inv/total_loss', inv_total_loss, step)
            L.log('train_inv/invariance_loss', inv_sim_loss, step)
            L.log('train_inv/variance_loss', inv_var_loss, step)
            L.log('train_inv/covariance_loss', inv_cov_loss, step)
            L.log('train_total/critic_inv_loss', update_loss, step)

        self.critic_optimizer.zero_grad()
        update_loss.backward()
        self.critic_optimizer.step()

    def update_actor_and_alpha(self, obs_masked, obs_aug_masked, L=None, step=None, update_alpha=True):
        mu, pi, log_pi, log_std = self.actor(obs_masked, detach=False)
        actor_Q1, actor_Q2 = self.critic(obs_masked, pi, detach=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss_orig = (self.alpha.detach() * log_pi - actor_Q).mean()

        aug_mu, aug_pi, aug_log_pi, aug_log_std = self.actor(obs_aug_masked, detach=False)
        actor_Q1_aug, actor_Q2_aug = self.critic(obs_aug_masked, aug_pi, detach=True)
        actor_Q_aug = torch.min(actor_Q1_aug, actor_Q2_aug)
        actor_loss_aug = (self.alpha.detach() * aug_log_pi - actor_Q_aug).mean()

        actor_loss = 0.5 * (actor_loss_orig + actor_loss_aug)

        std = log_std.exp()
        std = torch.max(std, torch.ones_like(std) * 1e-3)
        cov_mat = torch.diag_embed(std).detach()
        target_distribution = MultivariateNormal(mu, cov_mat)

        aug_log_std = torch.clamp(aug_log_std, -10.0, 2.0)
        aug_std = aug_log_std.exp()
        aug_std = torch.max(aug_std, torch.ones_like(aug_std) * 1e-3)
        aug_cov_mat = torch.diag_embed(aug_std)
        current_distribution = MultivariateNormal(aug_mu, aug_cov_mat)
        kl_loss = torch.distributions.kl_divergence(target_distribution, current_distribution).mean()

        if L is not None:
            L.log('train_actor/loss_orig', actor_loss_orig, step)
            L.log('train_actor/loss_aug', actor_loss_aug, step)
            L.log('train_actor/loss', actor_loss, step)
            L.log('train/kl_loss', kl_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)

        actor_loss += self.policy_consistency_weight * kl_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, L, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample_svea()
        with torch.no_grad():
            obs_aug1 = random_overlay(obs.clone())
            obs_cat = torch.cat([obs, obs_aug1, next_obs], dim=0)
            obs_cat_masked = self.apply_mask(obs_cat)
            batch_size = obs.shape[0]
            obs_masked = obs_cat_masked[:batch_size]
            obs_aug1_masked = obs_cat_masked[batch_size:2*batch_size]
            next_obs_masked = obs_cat_masked[2*batch_size:]

        self.update_critic(obs, obs_masked, obs_aug1, obs_aug1_masked, action, reward, next_obs_masked, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs_masked, obs_aug1_masked, L, step)

        if step % self.critic_target_update_freq == 0:
            self.soft_update_critic_target()

    def select_action(self, obs, test_env=False):
        _obs = self._obs_to_input(obs)
        _obs = self.apply_mask(_obs, test_env)
        with torch.no_grad():
            mu, _, _, _ = self.actor(_obs, compute_pi=False, compute_log_pi=False)
        return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        _obs = self._obs_to_input(obs)
        _obs = self.apply_mask(_obs)
        with torch.no_grad():
            mu, pi, _, _ = self.actor(_obs, compute_log_pi=False)
        return pi.cpu().data.numpy().flatten()
