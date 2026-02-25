import numpy as np
import torch
import torch.nn.functional as F

import utils
from algorithms.madi import MaDi
from augmentations import strong_augment


class MaDiCompareOnline(MaDi):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)

		self.gt_supervision_steps = 50 * int(args.episode_length)
		self.gt_start_step = int(getattr(args, "init_steps", 0))
		self.gt_end_step = int(self.gt_start_step) + int(self.gt_supervision_steps)
		self.gt_supervision_weight = 1.0
		self._last_gt_step = None
		self.gt_used = 0

		self._online_gt_frame = None
		self._online_gt_mask = None

	def set_online_gt(self, frame, mask):
		self._online_gt_frame = frame
		self._online_gt_mask = mask

	def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
		with torch.no_grad():
			next_obs = self.apply_mask(next_obs)
			_, policy_action, log_pi, _ = self.actor(next_obs)
			target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
			target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
			target_Q = reward + (not_done * self.discount * target_V)

		obs_aug = strong_augment(obs, self.augment, self.overlay_alpha)

		if self.svea_alpha == self.svea_beta:
			obs = utils.cat(obs, obs_aug)
			obs = self.apply_mask(obs)
			action = utils.cat(action, action)
			target_Q = utils.cat(target_Q, target_Q)
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = (self.svea_alpha + self.svea_beta) * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
		else:
			obs = self.apply_mask(obs)
			current_Q1, current_Q2 = self.critic(obs, action)
			critic_loss = self.svea_alpha * \
				(F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q))
			obs_aug = self.apply_mask(obs_aug)
			current_Q1_aug, current_Q2_aug = self.critic(obs_aug, action)
			critic_loss += self.svea_beta * \
				(F.mse_loss(current_Q1_aug, target_Q) + F.mse_loss(current_Q2_aug, target_Q))

		gt_loss = None
		gt_available = self._online_gt_frame is not None and self._online_gt_mask is not None
		gt_enabled = (
			gt_available
			and step is not None
			and step >= self.gt_start_step
			and step < self.gt_end_step
			and self._last_gt_step != step
		)
		if gt_enabled:
			gt_frame = self._online_gt_frame
			gt_mask = self._online_gt_mask
			if gt_mask.ndim == 3:
				gt_mask = gt_mask.unsqueeze(1)
			pred_masks = self.masker(gt_frame)
			gt_loss = F.binary_cross_entropy(pred_masks, gt_mask)
			critic_loss = critic_loss + self.gt_supervision_weight * gt_loss
			self._last_gt_step = step
			self.gt_used += int(pred_masks.shape[0])

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			L.log('train_gt/available', float(gt_available), step)
			L.log('train_gt/enabled', float(gt_enabled), step)
			L.log('train_gt/supervision_steps', float(self.gt_supervision_steps), step)
			L.log('train_gt/start_step', float(self.gt_start_step), step)
			L.log('train_gt/end_step', float(self.gt_end_step), step)
			L.log('train_gt/used', float(self.gt_used), step)
			if gt_loss is not None:
				L.log('train_masker/gt_loss', gt_loss, step)

		self.masker_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		self.masker_optimizer.step()
