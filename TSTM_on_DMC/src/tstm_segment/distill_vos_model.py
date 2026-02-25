"""
Knowledge distillation for temporal segmentation: train a small student from a larger teacher.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
import argparse
import time
import sys
sys.path.insert(0, os.path.dirname(__file__))

from tstm_segment.temporal_segmentation_network import SimpleCNN_ConvLSTM, bce_dice_loss
from stage2_train_vos_model import TemporalSegDataset, compute_iou_dice_from_logits
from augmentations import random_overlay

os.environ['DMCGB_DATASETS'] = "/data/duweicheng/tstm/TSTM_on_DMC/dataset"


class DistillationTrainer:
    """Temporal segmentation distillation trainer"""
    def __init__(self, teacher_model, student_model, device='cuda', 
                 learning_rate=1e-3, distill_temp=4.0, distill_alpha=0.7,
                 use_overlay=False, overlay_alpha=0.5, overlay_prob=0.5):
        self.teacher = teacher_model.to(device)
        self.student = student_model.to(device)
        self.device = device
        # freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
        
        # student optimizer
        self.optimizer = optim.Adam(self.student.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.temp = distill_temp
        self.alpha = distill_alpha
        self.use_overlay = use_overlay
        self.overlay_alpha = overlay_alpha
        self.overlay_prob = overlay_prob
        self.hard_criterion = bce_dice_loss
        self.train_losses = []
        self.val_losses = []
    
    def distillation_loss(self, student_logits, teacher_logits, temperature):
        """Distillation loss with temperature scaling (MSE on sigmoid outputs)."""
        student_soft = torch.sigmoid(student_logits / temperature)
        teacher_soft = torch.sigmoid(teacher_logits / temperature)
        loss = F.mse_loss(student_soft, teacher_soft)
        loss = loss * (temperature ** 2)
        
        return loss
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch."""
        self.student.train()
        self.teacher.eval()
        
        epoch_loss = 0.0
        epoch_hard_loss = 0.0
        epoch_soft_loss = 0.0
        
        for frames, masks in train_loader:
            frames = frames.to(self.device)
            masks = masks.to(self.device)
            
            # apply overlay augmentation (consistent with teacher training)
            if self.use_overlay and np.random.rand() < self.overlay_prob:
                try:
                    # prepare input layout
                    if frames.dim() == 4:
                        frames_input = frames
                    else:
                        B, T, C, H, W = frames.shape
                        frames_input = frames.permute(0, 2, 1, 3, 4).reshape(B, C*T, H, W)
                    
                    # apply overlay (expects 0-255)
                    frames_input = frames_input * 255.0
                    frames_aug = random_overlay(frames_input)
                    frames_aug = frames_aug / 255.0
                    
                    # convert back to original layout
                    if frames.dim() == 4:
                        frames = frames_aug
                    else:
                        frames = frames_aug.reshape(B, C, T, H, W).permute(0, 2, 1, 3, 4)
                except RuntimeError:
                    # last batch may fail overlay due to size mismatch, skip augmentation
                    pass
            
            # teacher forward (soft labels)
            with torch.no_grad():
                teacher_logits, _ = self.teacher(frames)
            # student forward
            student_logits, _ = self.student(frames)
            # hard loss vs. ground truth
            hard_loss = self._compute_loss(student_logits, masks, self.hard_criterion)
            # soft distillation loss vs. teacher
            soft_loss = self._compute_distill_loss(student_logits, teacher_logits)
            # combine losses
            loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            epoch_hard_loss += hard_loss.item()
            epoch_soft_loss += soft_loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        avg_hard = epoch_hard_loss / len(train_loader)
        avg_soft = epoch_soft_loss / len(train_loader)
        
        self.train_losses.append(avg_loss)
        
        print(f"\nEpoch {epoch} average losses:")
        print(f"  Total: {avg_loss:.4f}")
        print(f"  Hard (GT):  {avg_hard:.4f}")
        print(f"  Soft (Teacher): {avg_soft:.4f}")
        
        return avg_loss
    
    def _compute_loss(self, pred_masks, masks, criterion):
        """Compute loss for both sequence and single-frame inputs."""
        if pred_masks.dim() == 5:  # [B, T, 1, H, W]
            if masks.dim() == 4:  # [B, 1, H, W] -> [B, 1, 1, H, W]
                masks = masks.unsqueeze(1)
            elif masks.dim() == 3:  # [B, H, W] -> [B, 1, 1, H, W]
                masks = masks.unsqueeze(1).unsqueeze(2)
            T = pred_masks.shape[1]
            loss = sum(criterion(pred_masks[:, t], masks[:, t]) for t in range(T)) / T
        else:  # [B, 1, H, W]
            if masks.dim() == 3:  # [B, H, W] -> [B, 1, H, W]
                masks = masks.unsqueeze(1)
            loss = criterion(pred_masks, masks)
        return loss
    
    def _compute_distill_loss(self, student_logits, teacher_logits):
        """Compute distillation loss for both sequence and single-frame inputs."""
        if student_logits.dim() == 5:  # [B, T, 1, H, W]
            T = student_logits.shape[1]
            loss = sum(
                self.distillation_loss(student_logits[:, t], teacher_logits[:, t], self.temp)
                for t in range(T)
            ) / T
        else:  # [B, 1, H, W]
            loss = self.distillation_loss(student_logits, teacher_logits, self.temp)
        return loss
    
    def validate(self, val_loader):
        """Validate the student model."""
        self.student.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_dice = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for frames, masks in val_loader:
                frames = frames.to(self.device)
                masks = masks.to(self.device)
                student_logits, _ = self.student(frames)
                loss = self._compute_loss(student_logits, masks, self.hard_criterion)
                val_loss += loss.item()
                iou, dice = compute_iou_dice_from_logits(student_logits, masks)
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
        """Full training loop."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss, val_iou, val_dice = self.validate(val_loader)
            self.scheduler.step(val_loss)
            epoch_time = time.time() - epoch_start
            
            print(
                f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"Val IoU: {val_iou:.4f}, Val Dice: {val_dice:.4f}, Epoch Time: {epoch_time:.1f}s"
            )
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                
                save_dict = {
                    'model_state_dict': self.student.state_dict(),
                    'val_loss': val_loss,
                    'val_iou': val_iou,
                    'val_dice': val_dice,
                    'epoch': epoch,
                    'hidden_dim': self.student.hidden_dim,
                }
                
                torch.save(save_dict, save_dir / 'best_model.pth')
                print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
            
        print("Distillation complete")
        print(f"Best val loss: {best_val_loss:.4f}")
        print(f"Model saved to: {save_dir}")


def _benchmark_inference(model, device, batch_size: int, sequence_length: int, iters: int = 50, warmup: int = 10):
    model.eval()
    x = torch.rand(batch_size, sequence_length, 3, 84, 84, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    dt = (t1 - t0) / max(iters, 1)
    frames = batch_size * sequence_length
    fps = frames / dt if dt > 0 else float("inf")
    return float(dt), float(fps)


def main():
    parser = argparse.ArgumentParser(description='Temporal segmentation distillation')
    
    # data
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory (stage 1 output)')
    parser.add_argument('--teacher_path', type=str, required=True, help='Teacher checkpoint path')
    
    # teacher/student config
    parser.add_argument('--teacher_hidden', type=int, default=256, help='Teacher hidden_dim')
    parser.add_argument('--student_hidden', type=int, default=32, help='Student hidden_dim')
    # training params
    parser.add_argument('--sequence_length', type=int, default=5, help='Sequence length (5)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split ratio')
    # distillation params
    parser.add_argument('--distill_temp', type=float, default=4.0, help='Temperature')
    parser.add_argument('--distill_alpha', type=float, default=0.7, help='Alpha (teacher vs GT tradeoff)')
    # augmentation params
    parser.add_argument('--use_overlay', action='store_true', help='Use overlay augmentation')
    parser.add_argument('--overlay_alpha', type=float, default=0.5, help='Overlay alpha')
    parser.add_argument('--overlay_prob', type=float, default=0.5, help='Overlay probability')
    parser.add_argument('--benchmark', action='store_true', help='Benchmark teacher/student inference speed (once, before training)')
    parser.add_argument('--bench_batch_size', type=int, default=1, help='Benchmark batch size (B)')
    parser.add_argument('--bench_iters', type=int, default=50, help='Benchmark iterations')

    # misc
    parser.add_argument('--save_dir', type=str, default='logs/temporal_distilled', help='Model save directory')
    parser.add_argument('--num_workers', type=int, default=16, help='Dataloader workers')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_dataset = TemporalSegDataset(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length
    )
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    print(f"\nLoading teacher: {args.teacher_path}")
    try:
        checkpoint = torch.load(args.teacher_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(args.teacher_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    teacher_hidden = checkpoint.get('hidden_dim', args.teacher_hidden)
    if teacher_hidden not in (32, 256):
        raise ValueError(f"Unsupported teacher hidden_dim: {teacher_hidden}. Expected 32 or 256.")
    teacher = SimpleCNN_ConvLSTM(
        input_channels=3,
        num_classes=1,
        hidden_dim=teacher_hidden,
        kernel_size=3,
    ).to(device)
    teacher.load_state_dict(state_dict)
    
    teacher.eval()
    print(f"✓ Teacher loaded (hidden={teacher_hidden}, params={sum(p.numel() for p in teacher.parameters()):,})")

    if args.student_hidden not in (32, 256):
        raise ValueError(f"Unsupported student hidden_dim: {args.student_hidden}. Expected 32 or 256.")
    print(f"\nCreate student (hidden_dim={args.student_hidden})")
    student = SimpleCNN_ConvLSTM(
        input_channels=3,
        num_classes=1,
        hidden_dim=args.student_hidden,
        kernel_size=3,
    ).to(device)
    
    print(f"✓ Student created (params={sum(p.numel() for p in student.parameters()):,})")
    print(f"  Compression: {sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student.parameters()):.1f}x")

    if args.benchmark:
        dt_t, fps_t = _benchmark_inference(
            teacher,
            device=device,
            batch_size=int(args.bench_batch_size),
            sequence_length=int(args.sequence_length),
            iters=int(args.bench_iters),
        )
        dt_s, fps_s = _benchmark_inference(
            student,
            device=device,
            batch_size=int(args.bench_batch_size),
            sequence_length=int(args.sequence_length),
            iters=int(args.bench_iters),
        )
        speedup = (dt_t / dt_s) if dt_s > 0 else float("inf")
        print("\nInference benchmark")
        print(f"  Device: {device}")
        print(f"  Input: B={int(args.bench_batch_size)}, T={int(args.sequence_length)}, 84x84")
        print(f"  Teacher: {dt_t*1000:.2f} ms/batch, {fps_t:.1f} frames/s")
        print(f"  Student: {dt_s*1000:.2f} ms/batch, {fps_s:.1f} frames/s")
        print(f"  Speedup: {speedup:.2f}x (higher is faster)")

    trainer = DistillationTrainer(
        teacher_model=teacher,
        student_model=student,
        device=device,
        learning_rate=args.learning_rate,
        distill_temp=args.distill_temp,
        distill_alpha=args.distill_alpha,
        use_overlay=args.use_overlay,
        overlay_alpha=args.overlay_alpha,
        overlay_prob=args.overlay_prob
    )

    trainer.train(train_loader, val_loader, args.num_epochs, args.save_dir)
    
    print("\n✓ Training finished")
    print(f"Student saved: {args.save_dir}/best_model.pth")


if __name__ == '__main__':
    main()

