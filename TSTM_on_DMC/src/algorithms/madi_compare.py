import os
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import utils
from algorithms.madi import MaDi
from augmentations import strong_augment


class _GTMaskSampler(object):
	def __init__(self, data_dir, device, max_items=None, seed=0):
		self.data_dir = data_dir
		self.device = device
		self.max_items = max_items
		self.used = 0

		metadata_path = os.path.join(self.data_dir, "episodes_metadata.pkl")
		with open(metadata_path, "rb") as f:
			metadata = pickle.load(f)

		items = []
		for ep in metadata:
			for frame_info in ep.get("frames", []):
				frame_path = frame_info.get("frame_path")
				mask_path = frame_info.get("mask_path")
				if frame_path is None or mask_path is None:
					continue
				if not os.path.isabs(frame_path):
					frame_path = os.path.join(self.data_dir, frame_path)
				if not os.path.isabs(mask_path):
					mask_path = os.path.join(self.data_dir, mask_path)
				items.append((frame_path, mask_path))

		if self.max_items is not None and self.max_items > 0:
			items = items[: self.max_items]
		self.items = items

		self._rng = random.Random(seed)
		self._perm = list(range(len(self.items)))
		self._rng.shuffle(self._perm)
		self._ptr = 0

	@property
	def size(self):
		return len(self.items)

	def sample(self, batch_size):
		if self.size == 0:
			return None

		frames = []
		masks = []
		for _ in range(batch_size):
			if self._ptr >= len(self._perm):
				self._rng.shuffle(self._perm)
				self._ptr = 0

			idx = self._perm[self._ptr]
			self._ptr += 1
			frame_path, mask_path = self.items[idx]

			img = cv2.imread(frame_path)
			if img is None:
				continue
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img_t = torch.as_tensor(img, device=self.device).permute(2, 0, 1).float()

			mask = np.load(mask_path)
			if mask.ndim == 2:
				mask = np.expand_dims(mask, axis=0)
			mask_t = torch.as_tensor(mask, device=self.device).float()
			if mask_t.max() > 1.0:
				mask_t = (mask_t > 0.0).float()

			frames.append(img_t)
			masks.append(mask_t)

		if len(frames) == 0:
			return None

		frames_t = torch.stack(frames, dim=0)
		masks_t = torch.stack(masks, dim=0)
		self.used += frames_t.shape[0]
		return frames_t, masks_t


class MaDiCompare(MaDi):
	def __init__(self, obs_shape, action_shape, args):
		super().__init__(obs_shape, action_shape, args)

		self.gt_supervision_steps = 50 * int(args.episode_length)
		self.gt_end_step = int(getattr(args, "init_steps", 0)) + int(self.gt_supervision_steps)
		self.gt_supervision_weight = 1.0
		self.gt_batch_size = 1
		self.gt_update_freq = 1
		self._last_gt_step = None

		max_items = getattr(args, "madi_gt_max_items", None)
		if isinstance(max_items, int) and max_items < 0:
			max_items = None

		project_root = Path(__file__).resolve().parents[2]
		data_dir = project_root / args.log_dir / "temporal_training_data_gt" / f"{args.domain_name}_{args.task_name}" / f"seed_{args.seed}"
		metadata_path = data_dir / "episodes_metadata.pkl"
		if not metadata_path.exists():
			raise FileNotFoundError(
				f"MaDiCompare requires GT masks collected by stage1. Missing: {metadata_path}"
			)

		with open(str(metadata_path), "rb") as f:
			metadata = pickle.load(f)
		total_steps = 0
		for ep in metadata:
			if "num_steps" in ep and ep["num_steps"] is not None:
				total_steps += int(ep["num_steps"])
			else:
				total_steps += len(ep.get("frames", []))
		self.gt_sampler = _GTMaskSampler(str(data_dir), torch.device("cuda"), max_items=max_items, seed=args.seed)
		if total_steps > 0:
			self.gt_supervision_steps = int(total_steps)
		if self.gt_sampler.size > 0:
			self.gt_supervision_steps = min(int(self.gt_supervision_steps), int(self.gt_sampler.size))
		self.gt_end_step = int(getattr(args, "init_steps", 0)) + int(self.gt_supervision_steps)

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
		gt_enabled = (
			self.gt_sampler is not None
			and step is not None
			and step < self.gt_end_step
			and self.gt_update_freq is not None
			and self.gt_update_freq > 0
			and step % self.gt_update_freq == 0
		)
		if gt_enabled and self._last_gt_step != step:
			gt_batch = self.gt_sampler.sample(self.gt_batch_size)
			if gt_batch is not None:
				gt_frames, gt_masks = gt_batch
				pred_masks = self.masker(gt_frames)
				gt_loss = F.binary_cross_entropy(pred_masks, gt_masks)
				critic_loss = critic_loss + self.gt_supervision_weight * gt_loss
				self._last_gt_step = step

		if L is not None:
			L.log('train_critic/loss', critic_loss, step)
			if self.gt_sampler is not None:
				L.log('train_gt/enabled', float(gt_enabled), step)
				L.log('train_gt/supervision_steps', float(self.gt_supervision_steps), step)
				L.log('train_gt/end_step', float(self.gt_end_step), step)
				L.log('train_gt/dataset_size', float(self.gt_sampler.size), step)
				L.log('train_gt/used', float(self.gt_sampler.used), step)
				if gt_loss is not None:
					L.log('train_masker/gt_loss', gt_loss, step)

		self.masker_optimizer.zero_grad()
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()
		self.masker_optimizer.step()
