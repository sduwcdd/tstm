"""
ablation experiment algorithm class - based on TSTM

contains three ablation experiment variants:
1. TSTM_NoSeg - remove segmentation model preprocessing
2. TSTM_NoVICReg - remove VICReg contrastive learning loss
3. TSTM_NoPolicyConsistency - remove policy consistency loss

Each class performs ablation by inheriting the parent and overriding key methods with minimal changes.
"""

import torch
import torch.nn.functional as F
import utils
from algorithms.tstm import TSTM
import numpy as np


# ============================================================================
# Ablation experiment 1: Remove segmentation model preprocessing
# ============================================================================
class TSTM_NoSeg(TSTM):
    """
    Ablation experiment: Remove segmentation model preprocessing
    
    Key modifications:
    - __init__: Do not load temporal segmentation model
    - apply_mask: Return raw observation directly, without segmentation
    - Other components (VICReg loss, policy consistency) remain unchanged
    """
    
    def __init__(self, obs_shape, action_shape, args):
        # initialize base TSTM components
        TSTM.__init__(self, obs_shape, action_shape, args)
        
        # TSTM parameters
        self.device = torch.device("cuda")
        self.policy_consistency_weight = getattr(args, 'policy_consistency_weight', 2.0)
        self.inv_loss_weight = getattr(args, 'INV_loss_weight', 0.1)
        self.masker_policy_consistency_weight = getattr(args, 'masker_policy_consistency_weight', 0.0)
        self.inv_lambda = getattr(args, 'INV_lambda', 1.25)
        self.inv_mu = getattr(args, 'INV_mu', 1.25)
        self.inv_rho = getattr(args, 'INV_rho', 0.005)
        self.inv_gamma = getattr(args, 'INV_gamma', 1.0)
        self.inv_warmup_steps = getattr(args, 'INV_warmup_steps', 0)
        
        # Do not use temporal segmentation model in this ablation
        self.temporal_model = None
    
    def _load_temporal_model(self, args):
        """Override: NoSeg does not load a temporal segmentation model."""
        self.temporal_model = None
    
    def apply_mask(self, obs, test_env=False):
        """Ablation: return raw observation (no segmentation)."""
        return obs


# ============================================================================
# Ablation experiment 2: Remove VICReg contrastive learning loss
# ============================================================================
class TSTM_NoVICReg(TSTM):
    """
    Ablation: remove INV/VICReg contrastive loss.
    - update_critic: no INV/VICReg term, keep critic loss only
    - other components (temporal segmentation, policy consistency) unchanged
    """
    
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
        
        if hasattr(self, 'projector') and self.projector is not None:
            # Remove the last parameter group (projector is the last added)
            if len(self.critic_optimizer.param_groups) > 1:
                self.critic_optimizer.param_groups.pop()
            del self.projector
            self.projector = None
    
    def update_critic(self, obs, obs_masked, obs_aug1, obs_aug1_masked, action, reward, next_obs_masked, not_done, L=None, step=None):
        """Ablation: remove INV/VICReg loss and keep critic loss only."""
        # Compute target Q
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs_masked)
            target_Q1, target_Q2 = self.critic_target(next_obs_masked, policy_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)
        
        # Simplified: single forward on concatenated batch; loss is sum of two MSEs
        obs_cat_masked = utils.cat(obs_masked, obs_aug1_masked)
        action = utils.cat(action, action)
        target_Q = utils.cat(target_Q, target_Q)
        current_Q1, current_Q2 = self.critic(obs_cat_masked, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        # No INV/VICReg term; update_loss equals critic_loss
        update_loss = critic_loss
        
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)
            L.log('train_total/critic_loss_no_vicreg', update_loss, step)
        # backprop and optimize
        self.critic_optimizer.zero_grad()
        update_loss.backward()
        self.critic_optimizer.step()


# ============================================================================
# Ablation experiment 3: Remove policy consistency loss
# ============================================================================
class TSTM_NoPolicyConsistency(TSTM):
    """
    Ablation: remove policy consistency loss (KL divergence).
    - update_actor_and_alpha: remove KL divergence term
    - other components (segmentation, INV/VICReg) unchanged
    """
    
    def __init__(self, obs_shape, action_shape, args):
        super().__init__(obs_shape, action_shape, args)
    
    def update_actor_and_alpha(self, obs_masked, obs_aug_masked, L=None, step=None, update_alpha=True):
        """Ablation: remove KL-divergence policy consistency term."""
        # actor loss on original observations
        mu, pi, log_pi, log_std = self.actor(obs_masked, detach=False)
        actor_Q1, actor_Q2 = self.critic(obs_masked, pi, detach=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss_orig = (self.alpha.detach() * log_pi - actor_Q).mean()
        
        # actor loss on augmented observations
        aug_mu, aug_pi, aug_log_pi, aug_log_std = self.actor(obs_aug_masked, detach=False)
        actor_Q1_aug, actor_Q2_aug = self.critic(obs_aug_masked, aug_pi, detach=True)
        actor_Q_aug = torch.min(actor_Q1_aug, actor_Q2_aug)
        actor_loss_aug = (self.alpha.detach() * aug_log_pi - actor_Q_aug).mean()
        
        # average the two losses
        actor_loss = 0.5 * (actor_loss_orig + actor_loss_aug)
        
        # no KL divergence term added (use base actor_loss)
        
        if L is not None:
            L.log('train_actor/loss_orig', actor_loss_orig, step)
            L.log('train_actor/loss_aug', actor_loss_aug, step)
            L.log('train_actor/loss_no_pc', actor_loss, step)
            entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        
        # optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # update alpha
        if update_alpha:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            if L is not None:
                L.log('train_alpha/loss', alpha_loss, step)
                L.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()
