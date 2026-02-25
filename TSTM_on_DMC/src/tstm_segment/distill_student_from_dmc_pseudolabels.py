'''
See Appendix(Optical flow): Experimental Model of Light-Flow Distillation for Students
'''
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tstm_segment.temporal_segmentation_network import SimpleCNN_ConvLSTM

import augmentations


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_hwc_uint8(x_chw: np.ndarray) -> np.ndarray:
    x = np.transpose(x_chw, (1, 2, 0))
    if x.dtype != np.uint8:
        x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _overlay_rgb_mask(rgb_hwc_u8: np.ndarray, mask_hw_u8: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    m = (mask_hw_u8 > 0).astype(np.float32)[..., None]
    rgb = rgb_hwc_u8.astype(np.float32)
    color = np.zeros_like(rgb)
    color[..., 0] = 255.0
    out = rgb * (1.0 - alpha * m) + color * (alpha * m)
    return np.clip(out, 0, 255).astype(np.uint8)


def _hstack(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    return np.concatenate(images, axis=1)


def _vstack(images: list[np.ndarray]) -> np.ndarray:
    if not images:
        raise ValueError("no images")
    return np.concatenate(images, axis=0)


def _load_temporal_model(ckpt_path: str, device: torch.device) -> SimpleCNN_ConvLSTM:
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden_dim = ckpt.get("hidden_dim", 32)
    model = SimpleCNN_ConvLSTM(input_channels=3, num_classes=1, hidden_dim=int(hidden_dim), kernel_size=3)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def _teacher_pseudolabel(teacher, obs_chw_u8: np.ndarray, frame_stack: int, device, threshold: float) -> np.ndarray:
    c, h, w = obs_chw_u8.shape
    if c != 3 * int(frame_stack):
        raise ValueError(f"obs channels {c} != 3*frame_stack")
    frames = obs_chw_u8.reshape(int(frame_stack), 3, h, w).astype(np.float32) / 255.0
    frames_b = torch.from_numpy(frames).unsqueeze(0).to(device)
    logits, _ = teacher(frames_b)
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float().cpu().numpy()[0]  # [T,1,H,W]
    return (pred[:, 0] > 0.5).astype(np.uint8)


def _random_shift_int(x: torch.Tensor, pad: int, sx=None, sy=None):
    if x.dim() != 4:
        raise ValueError(f"expected [B,C,H,W], got {tuple(x.shape)}")
    b, c, h, w = x.shape
    pad = int(pad)
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    if sx is None:
        sx = torch.randint(-pad, pad + 1, (b,), device=x.device, dtype=torch.int64)
    if sy is None:
        sy = torch.randint(-pad, pad + 1, (b,), device=x.device, dtype=torch.int64)
    out = torch.empty((b, c, h, w), device=x.device, dtype=x.dtype)
    for i in range(b):
        x0 = pad + int(sx[i].item())
        y0 = pad + int(sy[i].item())
        out[i] = padded[i, :, y0 : y0 + h, x0 : x0 + w]
    return out, sx, sy


def _bce_dice_loss_with_pos_weight(pred_logits, target, bce_weight: float, pos_weight: float):
    if target.dtype != torch.float32:
        target = target.float()
    pw = torch.tensor([float(pos_weight)], device=pred_logits.device, dtype=pred_logits.dtype)
    bce = F.binary_cross_entropy_with_logits(pred_logits, target, pos_weight=pw)

    probs = torch.sigmoid(pred_logits)
    probs_f = probs.flatten(1)
    tgt_f = target.flatten(1)
    inter = (probs_f * tgt_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + tgt_f.sum(dim=1)
    dice = (2.0 * inter + 1e-6) / (denom + 1e-6)
    dice_loss = 1.0 - dice.mean()
    return float(bce_weight) * bce + (1.0 - float(bce_weight)) * dice_loss


class DMCPseudoLabelDataset(Dataset):
    def __init__(self, data_dir: str, split: str, val_episodes: int, seed: int):
        self.data_dir = Path(data_dir)
        self.split = str(split)

        meta_path = self.data_dir / "metadata.pkl"
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing {meta_path}")
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        ep_ids = [int(ep["episode_idx"]) for ep in meta["episodes"]]
        if not ep_ids:
            raise ValueError("No episodes in metadata")

        rng = np.random.RandomState(int(seed))
        rng.shuffle(ep_ids)
        val_set = set(ep_ids[: int(val_episodes)])

        self.items = []
        self.obs_mm = {}
        self.mask_mm = {}

        for eid in ep_ids:
            obs_path = self.data_dir / "episodes" / f"episode_{eid:04d}_obs.npy"
            mask_path = self.data_dir / "episodes" / f"episode_{eid:04d}_mask.npy"
            if not obs_path.exists() or not mask_path.exists():
                continue

            in_val = eid in val_set
            if (self.split == "val" and not in_val) or (self.split == "train" and in_val):
                continue

            obs = np.load(str(obs_path), mmap_mode="r")
            mask = np.load(str(mask_path), mmap_mode="r")
            self.obs_mm[eid] = obs
            self.mask_mm[eid] = mask

            n = int(obs.shape[0])
            for s in range(n):
                self.items.append((eid, s))

        if not self.items:
            raise ValueError(f"No samples for split={self.split}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        eid, s = self.items[int(idx)]
        obs = np.array(self.obs_mm[eid][s], copy=True)
        mask = np.array(self.mask_mm[eid][s], copy=True)
        return obs, mask


def collect(args):
    os.environ.setdefault("MUJOCO_GL", "egl")

    out_dir = Path(args.out_dir)
    episodes_dir = out_dir / "episodes"
    _ensure_dir(episodes_dir)
    _ensure_dir(out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = _load_temporal_model(args.teacher_ckpt, device)

    from env.wrappers import make_env

    import gym

    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=int(args.seed),
        episode_length=int(args.episode_length),
        frame_stack=int(args.frame_stack),
        action_repeat=int(args.action_repeat),
        image_size=int(args.image_size),
        mode=str(args.mode),
        intensity=float(args.distracting_cs_intensity),
    )

    episodes_meta = []
    t0 = time.time()
    for ep in range(int(args.collect_episodes)):
        obs = env.reset()
        done = False
        obs_list = []
        mask_list = []
        step = 0
        while not done:
            obs_np = np.array(obs, copy=False)
            if obs_np.dtype != np.uint8:
                obs_np = obs_np.astype(np.uint8)
            pseudo = _teacher_pseudolabel(
                teacher,
                obs_np,
                frame_stack=int(args.frame_stack),
                device=device,
                threshold=float(args.teacher_threshold),
            )
            obs_list.append(obs_np)
            mask_list.append(pseudo)

            action = env.action_space.sample()
            obs, _, done, _ = env.step(action)
            step += 1

        obs_arr = np.stack(obs_list, axis=0)
        mask_arr = np.stack(mask_list, axis=0)

        obs_path = episodes_dir / f"episode_{ep:04d}_obs.npy"
        mask_path = episodes_dir / f"episode_{ep:04d}_mask.npy"
        np.save(str(obs_path), obs_arr)
        np.save(str(mask_path), mask_arr)

        episodes_meta.append({"episode_idx": int(ep), "num_steps": int(obs_arr.shape[0])})
        print(f"[collect] episode {ep:03d}/{int(args.collect_episodes)} steps={int(obs_arr.shape[0])}")

    meta = {
        "created_time": time.time(),
        "domain_name": str(args.domain_name),
        "task_name": str(args.task_name),
        "mode": str(args.mode),
        "seed": int(args.seed),
        "episode_length": int(args.episode_length),
        "collect_episodes": int(args.collect_episodes),
        "frame_stack": int(args.frame_stack),
        "action_repeat": int(args.action_repeat),
        "image_size": int(args.image_size),
        "teacher_ckpt": str(args.teacher_ckpt),
        "teacher_threshold": float(args.teacher_threshold),
        "episodes": episodes_meta,
    }
    with open(out_dir / "metadata.pkl", "wb") as f:
        pickle.dump(meta, f)

    print(f"[collect] done out_dir={out_dir} time={time.time()-t0:.1f}s")


def train(args):
    save_dir = Path(args.save_dir)
    _ensure_dir(save_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    with open(Path(args.data_dir) / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    frame_stack = int(meta["frame_stack"])

    if bool(args.use_overlay):
        augmentations._load_places(batch_size=1, image_size=int(meta["image_size"]))

    train_ds = DMCPseudoLabelDataset(args.data_dir, split="train", val_episodes=int(args.val_episodes), seed=int(args.seed))
    val_ds = DMCPseudoLabelDataset(args.data_dir, split="val", val_episodes=int(args.val_episodes), seed=int(args.seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=True,
        drop_last=False,
    )

    student = SimpleCNN_ConvLSTM(input_channels=3, num_classes=1, hidden_dim=int(args.student_hidden), kernel_size=3).to(device)
    opt = torch.optim.Adam(student.parameters(), lr=float(args.lr))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=5)

    best = float("inf")
    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        student.train()
        tr_loss = 0.0

        for obs_u8, mask_u8 in train_loader:
            obs = obs_u8.to(device, non_blocking=True).float() / 255.0  # [B,3T,H,W]
            mask = mask_u8.to(device, non_blocking=True).float()  # [B,T,H,W]

            if bool(args.apply_shift):
                obs, sx, sy = _random_shift_int(obs, pad=int(args.shift_pad))
                mask, _, _ = _random_shift_int(mask, pad=int(args.shift_pad), sx=sx, sy=sy)

            if bool(args.use_overlay) and obs.is_cuda:
                if float(args.overlay_prob) >= 1.0 or torch.rand((), device=device).item() < float(args.overlay_prob):
                    try:
                        obs = augmentations.random_overlay(obs * 255.0) / 255.0
                    except FileNotFoundError as e:
                        print(f"[warn] random_overlay failed: {e}")

            b, c, h, w = obs.shape
            t = int(frame_stack)
            frames = obs.reshape(b, t, 3, h, w)
            logits, _ = student(frames)

            if logits.dim() != 5:
                raise RuntimeError(f"Unexpected logits shape: {tuple(logits.shape)}")

            masks2 = mask.unsqueeze(2)  # [B,T,1,H,W]
            loss = 0.0
            for i in range(t):
                loss = loss + _bce_dice_loss_with_pos_weight(
                    logits[:, i],
                    masks2[:, i],
                    bce_weight=float(args.bce_weight),
                    pos_weight=float(args.pos_weight),
                )
            loss = loss / float(t)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss.item())

        tr_loss /= max(1, len(train_loader))

        student.eval()
        val_loss = 0.0
        with torch.no_grad():
            for obs_u8, mask_u8 in val_loader:
                obs = obs_u8.to(device, non_blocking=True).float() / 255.0
                mask = mask_u8.to(device, non_blocking=True).float()

                if bool(args.apply_shift):
                    obs, sx, sy = _random_shift_int(obs, pad=int(args.shift_pad))
                    mask, _, _ = _random_shift_int(mask, pad=int(args.shift_pad), sx=sx, sy=sy)

                if bool(args.use_overlay) and obs.is_cuda:
                    if float(args.overlay_prob) >= 1.0 or torch.rand((), device=device).item() < float(args.overlay_prob):
                        try:
                            obs = augmentations.random_overlay(obs * 255.0) / 255.0
                        except FileNotFoundError:
                            pass

                b, c, h, w = obs.shape
                t = int(frame_stack)
                frames = obs.reshape(b, t, 3, h, w)
                logits, _ = student(frames)
                masks2 = mask.unsqueeze(2)

                loss = 0.0
                for i in range(t):
                    loss = loss + _bce_dice_loss_with_pos_weight(
                        logits[:, i],
                        masks2[:, i],
                        bce_weight=float(args.bce_weight),
                        pos_weight=float(args.pos_weight),
                    )
                loss = loss / float(t)
                val_loss += float(loss.item())

        val_loss /= max(1, len(val_loader))
        sched.step(val_loss)

        dt = time.time() - t0
        print(f"[student] epoch {epoch:03d}/{int(args.epochs)} train {tr_loss:.4f} val {val_loss:.4f} {dt:.1f}s")

        ckpt_last = {"epoch": int(epoch), "model_state_dict": student.state_dict(), "hidden_dim": int(args.student_hidden)}
        torch.save(ckpt_last, save_dir / "student_last.pth")

        if val_loss < best:
            best = val_loss
            ckpt_best = {"epoch": int(epoch), "model_state_dict": student.state_dict(), "hidden_dim": int(args.student_hidden), "val_loss": float(val_loss)}
            torch.save(ckpt_best, save_dir / "best_model.pth")

    print(f"Done. Student checkpoint: {save_dir / 'best_model.pth'}")


def viz(args):
    data_dir = Path(args.data_dir)
    episodes_dir = data_dir / "episodes"
    if not episodes_dir.exists():
        raise FileNotFoundError(f"Missing {episodes_dir}")

    out_dir = Path(args.out_dir) if args.out_dir is not None else (data_dir / "vis")
    _ensure_dir(out_dir)

    with open(data_dir / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    t = int(meta["frame_stack"])

    episode_files = sorted(episodes_dir.glob("episode_*_obs.npy"))
    if not episode_files:
        raise ValueError(f"No episodes found under {episodes_dir}")

    max_eps = int(args.episodes)
    steps_per_episode = int(args.steps_per_episode)
    stride = int(args.stride)

    n_saved = 0
    for obs_path in episode_files[:max_eps]:
        stem = obs_path.name.replace("_obs.npy", "")
        mask_path = episodes_dir / f"{stem}_mask.npy"
        if not mask_path.exists():
            continue

        obs = np.load(str(obs_path), mmap_mode="r")  # [S,3T,H,W]
        mask = np.load(str(mask_path), mmap_mode="r")  # [S,T,H,W]

        s_max = int(obs.shape[0])
        take = min(steps_per_episode, max(0, (s_max + stride - 1) // stride))
        for i in range(take):
            s = i * stride
            if s >= s_max:
                break

            obs_s = np.array(obs[s], copy=False)
            mask_s = np.array(mask[s], copy=False)

            frames = obs_s.reshape(t, 3, obs_s.shape[1], obs_s.shape[2])
            raw_row = []
            ov_row = []
            for k in range(t):
                rgb = _to_hwc_uint8(frames[k])
                m = (mask_s[k] > 0).astype(np.uint8) * 255
                raw_row.append(rgb)
                ov_row.append(_overlay_rgb_mask(rgb, m, alpha=float(args.overlay_alpha)))

            grid = _vstack([_hstack(raw_row), _hstack(ov_row)])
            Image.fromarray(grid).save(out_dir / f"{stem}_step_{s:04d}.png")
            n_saved += 1

    print(f"[viz] saved {n_saved} pngs to {out_dir}")


def main():
    p = argparse.ArgumentParser(description="Distill student from DMC observations using teacher pseudo-labels (independent pipeline)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("collect")
    pc.add_argument("--teacher_ckpt", type=str, required=True)
    pc.add_argument("--out_dir", type=str, required=True)
    pc.add_argument("--domain_name", type=str, default="walker")
    pc.add_argument("--task_name", type=str, default="walk")
    pc.add_argument("--mode", type=str, default="train", choices=["train", "color_easy", "color_hard", "video_easy", "video_hard", "distracting_cs"])
    pc.add_argument("--seed", type=int, default=0)
    pc.add_argument("--episode_length", type=int, default=1000)
    pc.add_argument("--collect_episodes", type=int, default=50)
    pc.add_argument("--frame_stack", type=int, default=5)
    pc.add_argument("--action_repeat", type=int, default=4)
    pc.add_argument("--image_size", type=int, default=84)
    pc.add_argument("--distracting_cs_intensity", type=float, default=0.1)
    pc.add_argument("--teacher_threshold", type=float, default=0.01)

    pt = sub.add_parser("train")
    pt.add_argument("--data_dir", type=str, required=True)
    pt.add_argument("--save_dir", type=str, required=True)
    pt.add_argument("--seed", type=int, default=0)
    pt.add_argument("--student_hidden", type=int, default=32)
    pt.add_argument("--epochs", type=int, default=50)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--num_workers", type=int, default=16)
    pt.add_argument("--bce_weight", type=float, default=0.3)
    pt.add_argument("--pos_weight", type=float, default=5.0)
    pt.add_argument("--val_episodes", type=int, default=5)
    pt.add_argument("--apply_shift", action="store_true")
    pt.add_argument("--shift_pad", type=int, default=4)
    pt.add_argument("--use_overlay", action="store_true")
    pt.add_argument("--overlay_prob", type=float, default=1.0)

    pv = sub.add_parser("viz")
    pv.add_argument("--data_dir", type=str, required=True)
    pv.add_argument("--out_dir", type=str, default=None)
    pv.add_argument("--episodes", type=int, default=2)
    pv.add_argument("--steps_per_episode", type=int, default=8)
    pv.add_argument("--stride", type=int, default=10)
    pv.add_argument("--overlay_alpha", type=float, default=0.4)

    args = p.parse_args()
    if args.cmd == "collect":
        collect(args)
    elif args.cmd == "train":
        train(args)
    else:
        viz(args)


if __name__ == "__main__":
    main()
