'''
See the appendix for the experiment(Transfer), which employs the Davis dataset for pre-training.
'''
import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tstm_segment.temporal_segmentation_network import SimpleCNN_ConvLSTM, bce_dice_loss, dice_loss
from augmentations import _load_places, random_overlay


def _resize_pad_to_square(img: np.ndarray, target_size: int, is_mask: bool, mode: str = "pad") -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img

    if str(mode) == "stretch":
        interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        return cv2.resize(img, (target_size, target_size), interpolation=interp)

    scale = min(float(target_size) / float(h), float(target_size) / float(w))
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))
    interp = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    if resized.ndim == 2:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    else:
        out = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return out


def _bbox_from_binary_mask(mask_hw: np.ndarray) -> Tuple[int, int, int, int]:
    ys, xs = np.where(mask_hw > 0)
    if xs.size == 0 or ys.size == 0:
        return 0, 0, int(mask_hw.shape[1]), int(mask_hw.shape[0])
    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1
    return x0, y0, x1, y1


def _crop_square_with_pad(img: np.ndarray, cx: float, cy: float, side: int, pad_value) -> np.ndarray:
    h, w = img.shape[:2]
    side = max(1, int(side))
    x0 = int(round(cx - side / 2.0))
    y0 = int(round(cy - side / 2.0))
    x1 = x0 + side
    y1 = y0 + side

    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if img.ndim == 2:
        out = np.full((side, side), pad_value, dtype=img.dtype)
        out[dst_y0:dst_y1, dst_x0:dst_x1] = img[src_y0:src_y1, src_x0:src_x1]
        return out
    out = np.full((side, side, img.shape[2]), pad_value, dtype=img.dtype)
    out[dst_y0:dst_y1, dst_x0:dst_x1, :] = img[src_y0:src_y1, src_x0:src_x1, :]
    return out


def _pad_to_square(img: np.ndarray, target_size: int, pad_value) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_size and w == target_size:
        return img
    top = (target_size - h) // 2
    bottom = target_size - h - top
    left = (target_size - w) // 2
    right = target_size - w - left
    if img.ndim == 2:
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=int(pad_value))
    return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_value)


def _bce_dice_loss_with_pos_weight(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float,
    pos_weight: float,
) -> torch.Tensor:
    if float(pos_weight) == 1.0:
        return bce_dice_loss(pred_logits, target, bce_weight=float(bce_weight))
    pw = torch.tensor([float(pos_weight)], device=pred_logits.device, dtype=pred_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pw)
    d = dice_loss(pred_logits, target)
    return float(bce_weight) * bce + (1.0 - float(bce_weight)) * d


class DavisVOSDataset(Dataset):
    """DAVIS VOS dataset loader.

    Expected DAVIS root structure:
      DAVIS/
        JPEGImages/480p/<sequence>/*.jpg
        Annotations/480p/<sequence>/*.png
        ImageSets/2017/train.txt, val.txt (or 2016)

    Masks are converted to binary foreground (any id > 0).

    Output:
      frames: [T, 3, 84, 84] if T>1 else [3, 84, 84]
      masks:  [T, 1, 84, 84] if T>1 else [1, 84, 84]
    """

    def __init__(
        self,
        davis_root: str,
        split: str,
        sequence_length: int = 5,
        year: str = "2017",
        resolution: str = "480p",
        target_size: int = 84,
        resize_mode: str = "pad",
        crop_margin: float = 0.15,
    ):
        self.root = Path(davis_root)
        self.split = str(split)
        self.sequence_length = int(sequence_length)
        self.year = str(year)
        self.resolution = str(resolution)
        self.target_size = int(target_size)
        self.resize_mode = str(resize_mode)
        self.crop_margin = float(crop_margin)

        set_file = self.root / "ImageSets" / self.year / f"{self.split}.txt"
        if not set_file.exists():
            raise FileNotFoundError(f"Missing DAVIS set file: {set_file}")

        with open(set_file, "r") as f:
            self.sequences = [ln.strip() for ln in f.readlines() if ln.strip()]
        if not self.sequences:
            raise ValueError(f"No sequences found in {set_file}")

        self.samples: List[Tuple[List[Path], List[Path]]] = []
        for seq in self.sequences:
            frames_dir = self.root / "JPEGImages" / self.resolution / seq
            ann_dir = self.root / "Annotations" / self.resolution / seq

            frame_files = sorted(list(frames_dir.glob("*.jpg")))
            if not frame_files:
                frame_files = sorted(list(frames_dir.glob("*.png")))
            if not frame_files:
                continue

            ann_files = [ann_dir / f"{p.stem}.png" for p in frame_files]

            if self.sequence_length > 1:
                for i in range(len(frame_files) - self.sequence_length + 1):
                    self.samples.append(
                        (frame_files[i : i + self.sequence_length], ann_files[i : i + self.sequence_length])
                    )
            else:
                for i in range(len(frame_files)):
                    self.samples.append(([frame_files[i]], [ann_files[i]]))

        if not self.samples:
            raise ValueError(f"No valid DAVIS samples found under {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_frame(self, path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb = _resize_pad_to_square(img_rgb, target_size=self.target_size, is_mask=False, mode=self.resize_mode)
        img_rgb = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_rgb, (2, 0, 1))
        return img_chw

    def _load_mask(self, path: Path) -> np.ndarray:
        if not path.exists():
            ann = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        else:
            ann = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if ann is None:
                ann = np.zeros((self.target_size, self.target_size), dtype=np.uint8)
        ann = _resize_pad_to_square(ann, target_size=self.target_size, is_mask=True, mode=self.resize_mode)
        mask = (ann > 0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)
        return mask

    def _load_frame_raw(self, path: Path) -> np.ndarray:
        img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {path}")
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    def _load_ann_raw(self, path: Path, shape_hw: Tuple[int, int]) -> np.ndarray:
        if not path.exists():
            return np.zeros(shape_hw, dtype=np.uint8)
        ann = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if ann is None:
            return np.zeros(shape_hw, dtype=np.uint8)
        if ann.shape[:2] != shape_hw:
            ann = cv2.resize(ann, (int(shape_hw[1]), int(shape_hw[0])), interpolation=cv2.INTER_NEAREST)
        return ann

    def __getitem__(self, idx: int):
        frame_paths, mask_paths = self.samples[idx]

        if str(self.resize_mode) != "mask_crop":
            frames = [self._load_frame(p) for p in frame_paths]
            masks = [self._load_mask(p) for p in mask_paths]
            frames = np.stack(frames) if len(frames) > 1 else frames[0]
            masks = np.stack(masks) if len(masks) > 1 else masks[0]
            return torch.from_numpy(frames), torch.from_numpy(masks)

        raw_frames = [self._load_frame_raw(p) for p in frame_paths]
        h0, w0 = raw_frames[0].shape[:2]
        raw_anns = [self._load_ann_raw(p, (h0, w0)) for p in mask_paths]

        union = np.zeros((h0, w0), dtype=np.uint8)
        for ann in raw_anns:
            union |= (ann > 0).astype(np.uint8)

        x0, y0, x1, y1 = _bbox_from_binary_mask(union)
        bw = max(1, int(x1 - x0))
        bh = max(1, int(y1 - y0))
        side = int(max(bw, bh))
        margin_px = int(round(float(self.crop_margin) * float(side)))
        side = int(side + 2 * margin_px)
        cx = (float(x0) + float(x1)) * 0.5
        cy = (float(y0) + float(y1)) * 0.5

        proc_frames: List[np.ndarray] = []
        proc_masks: List[np.ndarray] = []
        for img_rgb, ann in zip(raw_frames, raw_anns):
            cropped_img = _crop_square_with_pad(img_rgb, cx=cx, cy=cy, side=side, pad_value=(0, 0, 0))
            cropped_ann = _crop_square_with_pad(ann, cx=cx, cy=cy, side=side, pad_value=0)

            if side < int(self.target_size):
                cropped_img = _pad_to_square(cropped_img, int(self.target_size), pad_value=(0, 0, 0))
                cropped_ann = _pad_to_square(cropped_ann, int(self.target_size), pad_value=0)
            elif side > int(self.target_size):
                cropped_img = cv2.resize(cropped_img, (int(self.target_size), int(self.target_size)), interpolation=cv2.INTER_LINEAR)
                cropped_ann = cv2.resize(cropped_ann, (int(self.target_size), int(self.target_size)), interpolation=cv2.INTER_NEAREST)

            img_f = cropped_img.astype(np.float32) / 255.0
            img_chw = np.transpose(img_f, (2, 0, 1))
            mask = (cropped_ann > 0).astype(np.float32)
            mask = np.expand_dims(mask, axis=0)
            proc_frames.append(img_chw)
            proc_masks.append(mask)

        frames = np.stack(proc_frames) if len(proc_frames) > 1 else proc_frames[0]
        masks = np.stack(proc_masks) if len(proc_masks) > 1 else proc_masks[0]

        return torch.from_numpy(frames), torch.from_numpy(masks)


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


@dataclass
class TrainStats:
    loss: float
    hard_loss: float
    soft_loss: float


class DistillationTrainer:
    def __init__(
        self,
        teacher: SimpleCNN_ConvLSTM,
        student: SimpleCNN_ConvLSTM,
        device: torch.device,
        lr: float,
        distill_temp: float,
        distill_alpha: float,
        use_overlay: bool = False,
        overlay_prob: float = 0.5,
        bce_weight: float = 0.5,
        pos_weight: float = 1.0,
    ):
        self.teacher = teacher.to(device)
        self.student = student.to(device)
        self.device = device

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.student.parameters(), lr=lr)
        self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.5, patience=5
        )

        self.temp = float(distill_temp)
        self.alpha = float(distill_alpha)
        self.use_overlay = bool(use_overlay)
        self.overlay_prob = float(overlay_prob)
        self.bce_weight = float(bce_weight)
        self.pos_weight = float(pos_weight)

    @staticmethod
    def _hard_loss(
        student_logits: torch.Tensor,
        masks: torch.Tensor,
        bce_weight: float,
        pos_weight: float,
    ) -> torch.Tensor:
        if student_logits.dim() == 5:
            if masks.dim() == 4:
                masks = masks.unsqueeze(1)
            elif masks.dim() == 3:
                masks = masks.unsqueeze(1).unsqueeze(2)
            t = student_logits.shape[1]
            return (
                sum(
                    _bce_dice_loss_with_pos_weight(
                        student_logits[:, i],
                        masks[:, i],
                        bce_weight=bce_weight,
                        pos_weight=pos_weight,
                    )
                    for i in range(t)
                )
                / t
            )
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        return _bce_dice_loss_with_pos_weight(student_logits, masks, bce_weight=bce_weight, pos_weight=pos_weight)

    def _soft_distill_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        def _distill(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            a = torch.sigmoid(a / self.temp)
            b = torch.sigmoid(b / self.temp)
            return F.mse_loss(a, b) * (self.temp**2)

        if student_logits.dim() == 5:
            t = student_logits.shape[1]
            return sum(_distill(student_logits[:, i], teacher_logits[:, i]) for i in range(t)) / t
        return _distill(student_logits, teacher_logits)

    def train_epoch(self, loader: DataLoader) -> TrainStats:
        self.student.train()

        total = 0.0
        total_hard = 0.0
        total_soft = 0.0

        for frames, masks in loader:
            frames = frames.to(self.device)
            masks = masks.to(self.device)

            if self.use_overlay and frames.is_cuda and np.random.rand() < self.overlay_prob:
                if frames.dim() == 4:
                    frames_input = frames
                else:
                    b, t, c, h, w = frames.shape
                    frames_input = frames.permute(0, 2, 1, 3, 4).reshape(b, c * t, h, w)
                frames_input = frames_input * 255.0
                frames_aug = random_overlay(frames_input)
                frames_aug = frames_aug / 255.0
                if frames.dim() == 4:
                    frames = frames_aug
                else:
                    frames = frames_aug.reshape(b, c, t, h, w).permute(0, 2, 1, 3, 4)

            with torch.no_grad():
                t_logits, _ = self.teacher(frames)

            s_logits, _ = self.student(frames)

            hard = self._hard_loss(s_logits, masks, bce_weight=self.bce_weight, pos_weight=self.pos_weight)
            soft = self._soft_distill_loss(s_logits, t_logits)
            loss = self.alpha * soft + (1.0 - self.alpha) * hard

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            total += float(loss.item())
            total_hard += float(hard.item())
            total_soft += float(soft.item())

        n = max(1, len(loader))
        return TrainStats(loss=total / n, hard_loss=total_hard / n, soft_loss=total_soft / n)

    @torch.no_grad()
    def validate(self, loader: DataLoader):
        self.student.eval()
        total = 0.0
        total_iou = 0.0
        total_dice = 0.0
        n = 0

        for frames, masks in loader:
            frames = frames.to(self.device)
            masks = masks.to(self.device)

            s_logits, _ = self.student(frames)
            loss = self._hard_loss(s_logits, masks, bce_weight=self.bce_weight, pos_weight=self.pos_weight)
            iou, dice = compute_iou_dice_from_logits(s_logits, masks)

            total += float(loss.item())
            total_iou += float(iou)
            total_dice += float(dice)
            n += 1

        denom = max(1, n)
        return total / denom, total_iou / denom, total_dice / denom


def _save_checkpoint(path: Path, model: SimpleCNN_ConvLSTM, epoch: int, metrics: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "hidden_dim": getattr(model, "hidden_dim", None),
    }
    save_dict.update(metrics)
    torch.save(save_dict, path)


def main():
    parser = argparse.ArgumentParser(description="Pretrain TSTM temporal segmentation on public VOS datasets")

    parser.add_argument("--dataset", type=str, default="davis2017", choices=["davis2016", "davis2017"])
    parser.add_argument("--data_root", type=str, required=True, help="DAVIS root directory")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--target_size", type=int, default=84)
    parser.add_argument("--resize_mode", type=str, default="pad", choices=["pad", "stretch", "mask_crop"])
    parser.add_argument("--crop_margin", type=float, default=0.15)

    parser.add_argument("--sequence_length", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--teacher_hidden", type=int, default=256, choices=[32, 256])
    parser.add_argument("--student_hidden", type=int, default=32, choices=[32, 256])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--distill_temp", type=float, default=4.0)
    parser.add_argument("--distill_alpha", type=float, default=0.7)
    parser.add_argument("--bce_weight", type=float, default=0.5)
    parser.add_argument("--pos_weight", type=float, default=1.0)

    parser.add_argument("--use_overlay", action="store_true")
    parser.add_argument("--overlay_prob", type=float, default=0.5)
    parser.add_argument("--overlay_warmup_epochs", type=int, default=0)

    parser.add_argument("--teacher_only", action="store_true")
    parser.add_argument("--student_only", action="store_true")
    parser.add_argument("--teacher_ckpt", type=str, default=None)

    parser.add_argument("--metric_threshold", type=float, default=0.5)
    parser.add_argument("--teacher_select_metric", type=str, default="val_loss", choices=["val_loss", "val_iou", "val_dice"])

    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if bool(args.teacher_only) and bool(args.student_only):
        raise ValueError("--teacher_only and --student_only cannot both be set")

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if bool(args.use_overlay):
        _load_places(batch_size=1, image_size=int(args.target_size))

    year = "2016" if args.dataset == "davis2016" else "2017"
    train_ds = DavisVOSDataset(
        davis_root=args.data_root,
        split="train",
        sequence_length=int(args.sequence_length),
        year=year,
        resolution=str(args.resolution),
        target_size=int(args.target_size),
        resize_mode=str(args.resize_mode),
        crop_margin=float(args.crop_margin),
    )
    val_ds = DavisVOSDataset(
        davis_root=args.data_root,
        split="val",
        sequence_length=int(args.sequence_length),
        year=year,
        resolution=str(args.resolution),
        target_size=int(args.target_size),
        resize_mode=str(args.resize_mode),
        crop_margin=float(args.crop_margin),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
    )

    teacher = SimpleCNN_ConvLSTM(input_channels=3, num_classes=1, hidden_dim=int(args.teacher_hidden), kernel_size=3)
    student = SimpleCNN_ConvLSTM(input_channels=3, num_classes=1, hidden_dim=int(args.student_hidden), kernel_size=3)

    # teacher pretrain (supervised on DAVIS masks)
    teacher = teacher.to(device)
    teacher_opt = torch.optim.Adam(teacher.parameters(), lr=float(args.lr))
    teacher_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(teacher_opt, mode="min", factor=0.5, patience=5)

    save_dir = Path(args.save_dir)
    best_teacher = float("-inf")

    print(f"Device: {device}")
    print(f"Dataset: DAVIS{year} ({args.resolution}), target_size={int(args.target_size)}")
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    print(f"Teacher hidden={int(args.teacher_hidden)}, Student hidden={int(args.student_hidden)}")

    if not bool(args.student_only):
        best_teacher = float("-inf")

        for epoch in range(1, int(args.epochs) + 1):
            t0 = time.time()

            teacher.train()
            tr_loss = 0.0
            for frames, masks in train_loader:
                frames = frames.to(device)
                masks = masks.to(device)

                if (
                    bool(args.use_overlay)
                    and int(epoch) > int(args.overlay_warmup_epochs)
                    and frames.is_cuda
                    and np.random.rand() < float(args.overlay_prob)
                ):
                    b, t, c, h, w = frames.shape
                    frames_input = frames.permute(0, 2, 1, 3, 4).reshape(b, c * t, h, w)
                    frames_input = frames_input * 255.0
                    frames_aug = random_overlay(frames_input)
                    frames_aug = frames_aug / 255.0
                    frames = frames_aug.reshape(b, c, t, h, w).permute(0, 2, 1, 3, 4)

                logits, _ = teacher(frames)
                if logits.dim() == 5:
                    if masks.dim() == 4:
                        masks = masks.unsqueeze(1)
                    elif masks.dim() == 3:
                        masks = masks.unsqueeze(1).unsqueeze(2)
                    tt = logits.shape[1]
                    loss = (
                        sum(
                            _bce_dice_loss_with_pos_weight(
                                logits[:, i],
                                masks[:, i],
                                bce_weight=float(args.bce_weight),
                                pos_weight=float(args.pos_weight),
                            )
                            for i in range(tt)
                        )
                        / tt
                    )
                else:
                    if masks.dim() == 3:
                        masks = masks.unsqueeze(1)
                    loss = _bce_dice_loss_with_pos_weight(
                        logits,
                        masks,
                        bce_weight=float(args.bce_weight),
                        pos_weight=float(args.pos_weight),
                    )

                teacher_opt.zero_grad()
                loss.backward()
                teacher_opt.step()
                tr_loss += float(loss.item())

            tr_loss /= max(1, len(train_loader))

            # val
            teacher.eval()
            val_loss = 0.0
            val_iou = 0.0
            val_dice = 0.0
            n = 0
            with torch.no_grad():
                for frames, masks in val_loader:
                    frames = frames.to(device)
                    masks = masks.to(device)
                    logits, _ = teacher(frames)
                    if logits.dim() == 5:
                        if masks.dim() == 4:
                            masks2 = masks.unsqueeze(1)
                        elif masks.dim() == 3:
                            masks2 = masks.unsqueeze(1).unsqueeze(2)
                        else:
                            masks2 = masks
                        tt = logits.shape[1]
                        loss = (
                            sum(
                                _bce_dice_loss_with_pos_weight(
                                    logits[:, i],
                                    masks2[:, i],
                                    bce_weight=float(args.bce_weight),
                                    pos_weight=float(args.pos_weight),
                                )
                                for i in range(tt)
                            )
                            / tt
                        )
                    else:
                        masks2 = masks.unsqueeze(1) if masks.dim() == 3 else masks
                        loss = _bce_dice_loss_with_pos_weight(
                            logits,
                            masks2,
                            bce_weight=float(args.bce_weight),
                            pos_weight=float(args.pos_weight),
                        )

                    iou, dice = compute_iou_dice_from_logits(logits, masks, threshold=float(args.metric_threshold))
                    val_loss += float(loss.item())
                    val_iou += float(iou)
                    val_dice += float(dice)
                    n += 1

            denom = max(1, n)
            val_loss /= denom
            val_iou /= denom
            val_dice /= denom

            teacher_sched.step(val_loss)

            dt = time.time() - t0
            print(
                f"[Teacher] Epoch {epoch:03d}/{int(args.epochs)} | "
                f"train {tr_loss:.4f} | val {val_loss:.4f} | IoU {val_iou:.4f} | Dice {val_dice:.4f} | {dt:.1f}s"
            )

            _save_checkpoint(
                save_dir / "teacher_last.pth",
                teacher,
                epoch=epoch,
                metrics={"train_loss": tr_loss, "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice},
            )

            if str(args.teacher_select_metric) == "val_loss":
                score = -float(val_loss)
            elif str(args.teacher_select_metric) == "val_iou":
                score = float(val_iou)
            else:
                score = float(val_dice)

            if score > best_teacher:
                best_teacher = score
                _save_checkpoint(
                    save_dir / "teacher_best.pth",
                    teacher,
                    epoch=epoch,
                    metrics={"train_loss": tr_loss, "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice},
                )

        if bool(args.teacher_only):
            print(f"Done. Teacher checkpoint: {save_dir / 'teacher_best.pth'}")
            return

    # student distill
    teacher_ckpt_path = Path(str(args.teacher_ckpt)) if args.teacher_ckpt is not None else (save_dir / "teacher_best.pth")
    teacher_ckpt = torch.load(teacher_ckpt_path, map_location=device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])

    trainer = DistillationTrainer(
        teacher=teacher,
        student=student,
        device=device,
        lr=float(args.lr),
        distill_temp=float(args.distill_temp),
        distill_alpha=float(args.distill_alpha),
        use_overlay=bool(args.use_overlay),
        overlay_prob=float(args.overlay_prob),
        bce_weight=float(args.bce_weight),
        pos_weight=float(args.pos_weight),
    )

    best_student = float("inf")
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        tr = trainer.train_epoch(train_loader)
        val_loss, val_iou, val_dice = trainer.validate(val_loader)
        trainer.sched.step(val_loss)
        dt = time.time() - t0

        print(
            f"[Student] Epoch {epoch:03d}/{int(args.epochs)} | "
            f"train {tr.loss:.4f} (hard {tr.hard_loss:.4f}, soft {tr.soft_loss:.4f}) | "
            f"val {val_loss:.4f} | IoU {val_iou:.4f} | Dice {val_dice:.4f} | {dt:.1f}s"
        )

        _save_checkpoint(
            save_dir / "student_last.pth",
            trainer.student,
            epoch=epoch,
            metrics={"train_loss": tr.loss, "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice},
        )

        if val_loss < best_student:
            best_student = val_loss
            _save_checkpoint(
                save_dir / "best_model.pth",
                trainer.student,
                epoch=epoch,
                metrics={"train_loss": tr.loss, "val_loss": val_loss, "val_iou": val_iou, "val_dice": val_dice},
            )

    print(f"Done. Student checkpoint (for RL): {save_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
