import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from pathlib import Path
import pickle
import argparse
import time
from tstm_segment.temporal_segmentation_network import (
    SimpleCNN_ConvLSTM,
    bce_dice_loss,
)
import sys
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from augmentations import random_overlay
os.environ['DMCGB_DATASETS'] = "/your_path_to_dmcgb_datasets/"

def _ensure_5d_masks(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 5:
        return x
    if x.dim() == 4:
        return x.unsqueeze(1)
    raise ValueError(f"expected mask tensor with dim 4 or 5, got {x.dim()}")

def compute_iou_dice_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5):
    logits_5d = _ensure_5d_masks(logits)
    target_5d = _ensure_5d_masks(target)
    probs = torch.sigmoid(logits_5d)
    pred = (probs >= threshold).float()
    tgt = (target_5d > 0.5).float()
    pred = pred.flatten(0, 1)
    tgt = tgt.flatten(0, 1)
    pred_f = pred.flatten(1)
    tgt_f = tgt.flatten(1)
    inter = (pred_f * tgt_f).sum(dim=1)
    union = pred_f.sum(dim=1) + tgt_f.sum(dim=1) - inter
    iou = (inter / (union + 1e-6)).mean().item()
    dice = ((2.0 * inter) / (pred_f.sum(dim=1) + tgt_f.sum(dim=1) + 1e-6)).mean().item()
    return float(iou), float(dice)

class TemporalSegDataset(Dataset):
    """Temporal segmentation dataset (optimized for GPU dataloading)."""
    def __init__(self, data_dir, sequence_length=1, transform=None, 
                 use_overlay=False, overlay_alpha=0.5, overlay_prob=0.5,
                 preload_to_memory=False):

        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.transform = transform  
        self.use_overlay = use_overlay
        self.overlay_alpha = overlay_alpha
        self.overlay_prob = overlay_prob
        self.preload_to_memory = preload_to_memory
        
        # Preload Places365 in main process (avoid multiprocessing issues)
        if self.use_overlay:
            from augmentations import _load_places
            # Preload Places365 with batch_size=1, image_size=84
            _load_places(batch_size=1, image_size=84)
            print("Places365 preloaded")
        
        # Load episode metadata
        metadata_path = self.data_dir / "episodes_metadata.pkl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
        
        with open(metadata_path, 'rb') as f:
            self.episode_data = pickle.load(f)
        
        # Build sample list
        self.samples = []
        self._build_samples()
        print(f"Dataset loaded: {len(self.samples)} samples")
        
        # Optional: preload all data into memory (speed up I/O)
        self.memory_cache = {}
        if self.preload_to_memory:
            print("Preloading all data into memory...")
            self._preload_all_data()
            print(f"Preload done: {len(self.memory_cache)} files")
    
    def _preload_all_data(self):
        """Preload all frames and masks into memory"""
        from tqdm import tqdm
        
        unique_files = set()
        for sample in self.samples:
            for frame_info in sample['frames']:
                frame_path = frame_info['frame_path']
                episode_idx = sample['episode_idx']
                step = frame_info['step']
                
                unique_files.add(('frame', frame_path))
                unique_files.add(('mask', episode_idx, step))
        
        print(f"  Files to load: {len(unique_files)}")
        
        for file_info in tqdm(unique_files, desc='Preloading'):
            if file_info[0] == 'frame':
                frame_path = file_info[1]
                if frame_path not in self.memory_cache:
                    self.memory_cache[frame_path] = self._load_frame(frame_path)
            else:  # mask
                episode_idx, step = file_info[1], file_info[2]
                mask_key = f"mask_{episode_idx}_{step}"
                if mask_key not in self.memory_cache:
                    self.memory_cache[mask_key] = self._load_mask(episode_idx, step)
    
    def _build_samples(self):
        """
        Build samples using sliding window over each episode.
        For sequence_length=5, each sample predicts the last frame's mask.
        """
        for episode in self.episode_data:
            episode_idx = episode['episode_idx']
            frames = episode['frames']
            
            # Sequence data (sliding window)
            if self.sequence_length > 1:
                # Slide with stride=1
                for i in range(len(frames) - self.sequence_length + 1):
                    sequence = frames[i:i + self.sequence_length]
                    self.samples.append({
                        'episode_idx': episode_idx,
                        'frames': sequence,
                        'is_sequence': True,
                        'target_frame_idx': i + self.sequence_length - 1  # index of last frame
                    })
            else:
                # Single frame
                for idx, frame in enumerate(frames):
                    self.samples.append({
                        'episode_idx': episode_idx,
                        'frames': [frame],
                        'is_sequence': False,
                        'target_frame_idx': idx
                    })
    
    def _load_frame(self, frame_path):
        """Load a single RGB frame (with optional memory cache)."""
        # Return from cache if preloaded
        if self.preload_to_memory and frame_path in self.memory_cache:
            return self.memory_cache[frame_path]
        
        image = cv2.imread(frame_path)
        if image is None:
            raise ValueError(f"Failed to load image: {frame_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        # Convert to CHW format
        image = np.transpose(image, (2, 0, 1))
        return image
    
    def _load_mask(self, episode_idx, step):
        """Load mask (.npy preferred, fallback to .pkl), with optional cache."""
        # Return from cache if preloaded
        mask_key = f"mask_{episode_idx}_{step}"
        if self.preload_to_memory and mask_key in self.memory_cache:
            return self.memory_cache[mask_key]
        
        # Prefer .npy format (SAM bbox-based masks)
        mask_path_npy = self.data_dir / "masks" / f"episode_{episode_idx:04d}_step_{step:04d}.npy"
        mask_path_pkl = self.data_dir / "masks" / f"episode_{episode_idx:04d}_step_{step:04d}.pkl"
        
        if mask_path_npy.exists():
            # Load binary mask from .npy
            mask = np.load(mask_path_npy).astype(np.float32)
            # Add channel dimension
            if mask.ndim == 2:
                mask = np.expand_dims(mask, axis=0)  # [1, H, W]
            return mask
        
        elif mask_path_pkl.exists():
            # Load .pkl format (legacy SAM)
            with open(mask_path_pkl, 'rb') as f:
                masks = pickle.load(f)
            
            if len(masks) == 0:
                return np.zeros((1, 84, 84), dtype=np.float32)
            
            # Select largest-area mask
            largest_mask = max(masks, key=lambda x: x['area'])
            mask = largest_mask['segmentation'].astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
            return mask
        
        else:
            # Mask not found -> zeros
            return np.zeros((1, 84, 84), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Return a single sample.
        Returns:
            frames: [T, C, H, W] or [C, H, W]
            masks: [T, 1, H, W] or [1, H, W]
        """
        sample = self.samples[idx]
        episode_idx = sample['episode_idx']
        frame_list = sample['frames']
        
        frames = []
        masks = []
        
        for frame_info in frame_list:
            frame_path = frame_info['frame_path']
            step = frame_info['step']
            
            # Load frame and mask
            frame = self._load_frame(frame_path)
            mask = self._load_mask(episode_idx, step)
            
            # Ensure size matches
            if frame.shape[1:] != mask.shape[1:]:
                # Resize mask to match frame if needed
                mask = cv2.resize(mask[0], (frame.shape[2], frame.shape[1]), 
                                interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, axis=0)
            
            frames.append(frame)
            masks.append(mask)
        
        frames = np.stack(frames) if len(frames) > 1 else frames[0]
        masks = np.stack(masks) if len(masks) > 1 else masks[0]
        
        # To tensor
        frames_tensor = torch.from_numpy(frames)
        masks_tensor = torch.from_numpy(masks)
        
        # Overlay is applied in TemporalTrainer.train_epoch (not here)
        
        # Backward-compatible transform support
        if self.transform is not None:
            frames_np = frames_tensor.numpy()
            masks_np = masks_tensor.numpy()
            frames_np, masks_np = self.transform(frames_np, masks_np)
            frames_tensor = torch.from_numpy(frames_np)
            masks_tensor = torch.from_numpy(masks_np)
        
        return frames_tensor, masks_tensor


class TemporalTrainer:
    """Trainer for temporal segmentation models."""
    
    def __init__(self, model, device='cuda', learning_rate=1e-4, 
                 use_overlay=False, overlay_alpha=0.5, overlay_prob=0.5):

        self.model = model.to(device)
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # LR scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Loss
        self.criterion = bce_dice_loss
        
        # Augmentation config
        self.use_overlay = use_overlay
        self.overlay_alpha = overlay_alpha
        self.overlay_prob = overlay_prob
        
        # History
        self.train_losses = []
        self.val_losses = []
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch."""
        self.model.train()
        epoch_loss = 0.0
        
        for frames, masks in train_loader:
            frames = frames.to(self.device)
            masks = masks.to(self.device)
            
            # Apply overlay augmentation
            if self.use_overlay and np.random.rand() < self.overlay_prob:
                # Prepare input format: random_overlay needs [B, C*T, H, W]
                if frames.dim() == 4:  # Single frame [B, C, H, W]
                    frames_input = frames
                else:  # Sequence [B, T, C, H, W]
                    B, T, C, H, W = frames.shape
                    frames_input = frames.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)
                
                # Overlay expects [0,255]
                frames_input = frames_input * 255.0
                frames_aug = random_overlay(frames_input)
                frames_aug = frames_aug / 255.0
                
                # Convert back
                if frames.dim() == 4:  # single frame
                    frames = frames_aug
                else:  # sequence
                    frames = frames_aug.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
            
            # Forward
            pred_masks, _ = self.model(frames)
            loss = self._compute_loss(pred_masks, masks)
            
            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Log
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def _compute_loss(self, pred_masks, masks):
        """Compute loss (supports both sequence and single-frame)."""
        # Sequence: average loss across frames
        if pred_masks.dim() == 5:  # [B, T, 1, H, W]
            if masks.dim() == 4:  # [B, 1, H, W] -> [B, 1, 1, H, W]
                masks = masks.unsqueeze(1)
            elif masks.dim() == 3:  # [B, H, W] -> [B, 1, 1, H, W]
                masks = masks.unsqueeze(1).unsqueeze(2)
            T = pred_masks.shape[1]
            loss = sum(self.criterion(pred_masks[:, t], masks[:, t]) for t in range(T)) / T
        # Single frame
        else:  # [B, 1, H, W]
            if masks.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
                masks = masks.unsqueeze(1)
            loss = self.criterion(pred_masks, masks)
        
        return loss
    
    def validate(self, val_loader):
        """Validate model."""
        self.model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for frames, masks in val_loader:
                frames = frames.to(self.device)
                masks = masks.to(self.device)
                pred_masks, _ = self.model(frames)
                loss = self._compute_loss(pred_masks, masks)
                val_loss += loss.item()
                iou, dice = compute_iou_dice_from_logits(pred_masks, masks)
                val_iou += iou
                val_dice += dice
                n_batches += 1
        
        denom = max(n_batches, 1)
        avg_loss = val_loss / denom
        avg_iou = val_iou / denom
        avg_dice = val_dice / denom
        self.val_losses.append(avg_loss)
        
        return avg_loss, avg_iou, avg_dice
    
    def train(self, train_loader, val_loader, num_epochs, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        best_val_loss = float('inf')
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_iou, val_dice = self.validate(val_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            old_lr = current_lr
            self.scheduler.step(val_loss)
            new_lr = self.optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val IoU:  {val_iou:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Learning Rate: {new_lr:.2e}")
            print(f"  Epoch Time: {epoch_time:.1f}s")
            if new_lr != old_lr:
                print(f"  Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")
            print("-" * 80)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'hidden_dim': getattr(self.model, 'hidden_dim', None),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'val_iou': val_iou,
                    'val_dice': val_dice,
                }, save_dir / 'best_model.pth')
                print(f"  Saved best model (Val Loss: {val_loss:.4f})")
            
            # Save final model
            torch.save({
                'epoch': num_epochs,
                'model_state_dict': self.model.state_dict(),
                'hidden_dim': getattr(self.model, 'hidden_dim', None),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
            }, save_dir / 'final_model.pth')
        
        print("Training completed")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved in: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train temporal segmentation model')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory (Stage 1 output)')
    parser.add_argument('--model_type', type=str, default='conv_lstm', choices=['conv_lstm'], help='Model type: conv_lstm only')
    parser.add_argument('--sequence_length', type=int, default=1, help='Sequence length (1=single frame, >1=video sequence)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=50, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--num_workers', type=int, default=8, help='Dataloader workers')
    parser.add_argument('--preload_to_memory', action='store_true', help='Preload all data into RAM (faster I/O, higher memory)')
    
    # SimpleCNN_ConvLSTM config
    parser.add_argument('--conv_lstm_hidden', type=int, default=256, choices=[32, 256], help='ConvLSTM hidden dim')
    parser.add_argument('--conv_lstm_kernel', type=int, default=3, help='Conv kernel size')
    
    # Augmentation (using project overlay)
    parser.add_argument('--use_overlay', action='store_true', help='Use overlay augmentation (sample backgrounds from Places365)')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Overlay mix alpha')
    parser.add_argument('--overlay_prob', type=float, default=0.5, help='Probability to apply overlay')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Stage 2: Train temporal segmentation model")
    print("=" * 80)
    print(f"Data dir: {args.data_dir}")
    print(f"Model type: {args.model_type}")
    print(f"Sequence length: {args.sequence_length}")
    print("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Data augmentation settings
    if args.use_overlay:
        print(f"Overlay augmentation enabled")
        print(f"  - Overlay prob: {args.overlay_prob}")
        print(f"  - Alpha: {args.overlay_alpha}")
        print(f"  - Backgrounds: Places365")
    
    # Create dataset
    full_dataset = TemporalSegDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        use_overlay=args.use_overlay,
        overlay_alpha=args.overlay_alpha,
        overlay_prob=args.overlay_prob,
        preload_to_memory=args.preload_to_memory
    )
    
    # Split train and validation sets
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        prefetch_factor=8,
        persistent_workers=True
    )
    
    # Create model (only support conv_lstm)
    model = SimpleCNN_ConvLSTM(
        input_channels=3,
        num_classes=1,
        hidden_dim=args.conv_lstm_hidden,
        kernel_size=args.conv_lstm_kernel,
    )
    
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    
    # Set save directory
    if args.save_dir is None:
        args.save_dir = Path(args.data_dir).parent / "temporal_model"
    
    # Create trainer and train
    trainer = TemporalTrainer(
        model, 
        device=device, 
        learning_rate=args.learning_rate,
        use_overlay=args.use_overlay,
        overlay_alpha=args.overlay_alpha,
        overlay_prob=args.overlay_prob
    )
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir
    )
    
    print("\n" + "=" * 80)
    print(f"Stage 2 complete. Model saved at: {args.save_dir}")
    print("=" * 80)
    print(f"\nNext: use the trained model for RL training")
    print(f"python src/train.py --algorithm tstm --temporal_model_path {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()

