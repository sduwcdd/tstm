'''
See the appendix for experiments(Transfer) demonstrating the visualisation of segmentation results from the trained segmentation model.(Optional)
'''
import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tstm_segment.pretrain_public_vos import DavisVOSDataset
from tstm_segment.temporal_segmentation_network import SimpleCNN_ConvLSTM

import augmentations


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def _to_uint8_rgb(chw01: np.ndarray) -> np.ndarray:
    hwc = np.transpose(chw01, (1, 2, 0))
    hwc = np.clip(hwc * 255.0, 0.0, 255.0).astype(np.uint8)
    return hwc


def _chw_to_uint8_rgb_auto(chw: np.ndarray) -> np.ndarray:
    """Convert CHW image to HWC uint8 RGB.

    Handles common DMC obs formats:
    - uint8 in [0, 255]
    - float in [0, 1]
    - float in [0, 255]
    """
    if chw.dtype == np.uint8:
        return np.transpose(chw, (1, 2, 0)).copy()

    x = chw.astype(np.float32, copy=False)
    vmax = float(np.max(x)) if x.size else 0.0
    if vmax <= 1.5:
        x = x * 255.0
    x = np.clip(x, 0.0, 255.0).astype(np.uint8)
    return np.transpose(x, (1, 2, 0)).copy()


def _normalize_chw_for_model(chw: np.ndarray) -> np.ndarray:
    """Convert CHW to float32 in [0,1] for the segmentation model."""
    if chw.dtype == np.uint8:
        return chw.astype(np.float32) / 255.0
    x = chw.astype(np.float32, copy=False)
    vmax = float(np.max(x)) if x.size else 0.0
    if vmax > 1.5:
        x = x / 255.0
    return x


def _mask_to_uint8(mask: np.ndarray) -> np.ndarray:
    if mask.ndim == 3 and mask.shape[0] == 1:
        mask = mask[0]
    return (mask > 0.5).astype(np.uint8)


def _overlay(image_rgb_u8: np.ndarray, mask_u8: np.ndarray, color=(0, 255, 255), alpha=0.35) -> np.ndarray:
    out = image_rgb_u8.copy()
    colored = np.zeros_like(out)
    colored[mask_u8 > 0] = np.array(color, dtype=np.uint8)
    out = cv2.addWeighted(out, 1.0 - alpha, colored, alpha, 0)
    return out


def _apply_mask_to_rgb(image_rgb_u8: np.ndarray, mask_u8: np.ndarray) -> np.ndarray:
    out = image_rgb_u8.copy()
    out[mask_u8 == 0] = 0
    return out


def _hstack(images: Tuple[np.ndarray, ...]) -> np.ndarray:
    return np.concatenate(list(images), axis=1)


def _predict_last_frame(
    model: SimpleCNN_ConvLSTM,
    obs_chw_u8_or_f: np.ndarray,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray]:
    c_total, h, w = obs_chw_u8_or_f.shape
    t = c_total // 3
    obs_seq = obs_chw_u8_or_f.reshape(t, 3, h, w)
    obs_seq_norm = np.stack([_normalize_chw_for_model(obs_seq[i]) for i in range(t)], axis=0)
    frames_b = torch.from_numpy(obs_seq_norm).float().unsqueeze(0).to(device)
    logits, _ = model(frames_b)
    probs = torch.sigmoid(logits)
    pred = (probs >= float(threshold)).float().cpu().numpy()[0]
    img_last = _chw_to_uint8_rgb_auto(obs_seq[-1])
    pr_last = (pred[-1, 0] > 0.5).astype(np.uint8)
    return img_last, pr_last


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
def visualize_davis_val(
    davis_root: str,
    davis_year: str,
    resolution: str,
    sequence_length: int,
    target_size: int,
    ckpt_path: str,
    out_dir: str,
    num_samples: int,
    threshold: float,
    seed: int,
):
    rng = np.random.RandomState(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_temporal_model(ckpt_path, device)

    ds = DavisVOSDataset(
        davis_root=davis_root,
        split="val",
        sequence_length=sequence_length,
        year=davis_year,
        resolution=resolution,
        target_size=target_size,
    )

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    idxs = rng.choice(len(ds), size=min(num_samples, len(ds)), replace=False)

    for k, idx in enumerate(idxs.tolist()):
        frames, masks = ds[int(idx)]

        if frames.dim() == 3:
            frames_b = frames.unsqueeze(0).unsqueeze(0)  # [1,1,3,H,W]
            masks_b = masks.unsqueeze(0).unsqueeze(0)    # [1,1,1,H,W]
        else:
            frames_b = frames.unsqueeze(0)  # [1,T,3,H,W]
            masks_b = masks.unsqueeze(0)    # [1,T,1,H,W]

        logits, _ = model(frames_b.to(device))
        probs = torch.sigmoid(logits)
        pred = (probs >= float(threshold)).float().cpu().numpy()

        frames_np = frames_b.cpu().numpy()[0]
        masks_np = masks_b.cpu().numpy()[0]
        pred_np = pred[0]

        T = frames_np.shape[0]
        for t in range(T):
            img_u8 = _to_uint8_rgb(frames_np[t])
            gt_u8 = _mask_to_uint8(masks_np[t])
            pr_u8 = _mask_to_uint8(pred_np[t])

            overlay_gt = _overlay(img_u8, gt_u8, color=(0, 255, 0), alpha=0.35)
            overlay_pr = _overlay(img_u8, pr_u8, color=(0, 255, 255), alpha=0.35)

            base = out_dir / f"sample_{k:04d}_idx_{int(idx):06d}_t_{t:02d}"
            cv2.imwrite(str(base.with_suffix(".rgb.png")), cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(base.with_suffix(".gt.png")), gt_u8 * 255)
            cv2.imwrite(str(base.with_suffix(".pred.png")), pr_u8 * 255)
            cv2.imwrite(str(base.with_suffix(".overlay_gt.png")), cv2.cvtColor(overlay_gt, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(base.with_suffix(".overlay_pred.png")), cv2.cvtColor(overlay_pr, cv2.COLOR_RGB2BGR))

    print(f"Saved DAVIS val visualizations to: {out_dir}")


@torch.no_grad()
def visualize_dmc_rollout(
    ckpt_path: str,
    out_dir: str,
    domain_name: str,
    task_name: str,
    seed: int,
    episode_length: int,
    frame_stack: int,
    action_repeat: int,
    image_size: int,
    env_mode: str,
    distracting_cs_intensity: float,
    steps: int,
    threshold: float,
    apply_rl_aug: bool,
    rl_shift_pad: int,
    rl_overlay_prob: float,
    save_every: int,
):
    os.environ.setdefault("MUJOCO_GL", "egl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _load_temporal_model(ckpt_path, device)

    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # Lazy import: env stack pulls in dm_control/glfw which warns if DISPLAY is missing.
    from env.wrappers import make_env

    import gym

    gym.logger.set_level(40)
    env = make_env(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        episode_length=episode_length,
        frame_stack=frame_stack,
        action_repeat=action_repeat,
        image_size=image_size,
        mode=env_mode,
        intensity=distracting_cs_intensity,
    )

    obs = env.reset()
    # LazyFrames -> np array
    obs_np = np.array(obs, copy=False)  # shape [3*frame_stack, H, W]

    for step in range(int(steps)):
        if int(save_every) > 0 and (step % int(save_every) == 0):
            img_raw_last, pr_raw_last = _predict_last_frame(model, obs_np, device, threshold)
            overlay_raw = _overlay(img_raw_last, pr_raw_last, color=(0, 255, 255), alpha=0.35)
            masked_raw = _apply_mask_to_rgb(img_raw_last, pr_raw_last)

            grid = _hstack((img_raw_last, overlay_raw, masked_raw))

            if apply_rl_aug:
                obs_t = torch.from_numpy(obs_np).unsqueeze(0).to(device)
                obs_t = obs_t.float() if obs_t.dtype != torch.float32 else obs_t
                obs_shift = augmentations.random_shift(obs_t, pad=int(rl_shift_pad))
                obs_shift_np = obs_shift.squeeze(0).detach().cpu().numpy()
                img_shift_last, pr_shift_last = _predict_last_frame(model, obs_shift_np, device, threshold)
                overlay_shift = _overlay(img_shift_last, pr_shift_last, color=(0, 255, 255), alpha=0.35)
                masked_shift = _apply_mask_to_rgb(img_shift_last, pr_shift_last)

                grid = _hstack((grid, img_shift_last, overlay_shift, masked_shift))

                obs_aug = obs_shift
                if device.type != "cuda":
                    if float(rl_overlay_prob) > 0.0:
                        print("[warn] --apply_rl_aug requested overlay, but CUDA is not available; skipping random_overlay")
                else:
                    if float(rl_overlay_prob) >= 1.0 or torch.rand((), device=device).item() < float(rl_overlay_prob):
                        try:
                            obs_aug = augmentations.random_overlay(obs_shift.clone())
                        except FileNotFoundError as e:
                            print(f"[warn] random_overlay failed (Places365 not found?): {e}")
                            obs_aug = obs_shift

                obs_aug_np = obs_aug.squeeze(0).detach().cpu().numpy()
                img_aug_last, pr_aug_last = _predict_last_frame(model, obs_aug_np, device, threshold)
                overlay_aug = _overlay(img_aug_last, pr_aug_last, color=(0, 255, 255), alpha=0.35)
                masked_aug = _apply_mask_to_rgb(img_aug_last, pr_aug_last)
                grid = _hstack((grid, img_aug_last, overlay_aug, masked_aug))

            base = out_dir / f"step_{step:05d}"
            cv2.imwrite(str(base.with_suffix(".grid.png")), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))

        action = env.action_space.sample()
        obs, _, done, _ = env.step(action)
        obs_np = np.array(obs, copy=False)
        if done:
            obs = env.reset()
            obs_np = np.array(obs, copy=False)

    print(f"Saved DMC rollout visualizations to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Visualize temporal segmentation predictions on DAVIS val and DMC")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_davis = sub.add_parser("davis", help="visualize on DAVIS val split")
    p_davis.add_argument("--data_root", type=str, required=True)
    p_davis.add_argument("--year", type=str, default="2017", choices=["2016", "2017"])
    p_davis.add_argument("--resolution", type=str, default="480p")
    p_davis.add_argument("--sequence_length", type=int, default=5)
    p_davis.add_argument("--target_size", type=int, default=84)
    p_davis.add_argument("--ckpt", type=str, required=True)
    p_davis.add_argument("--out_dir", type=str, required=True)
    p_davis.add_argument("--num_samples", type=int, default=50)
    p_davis.add_argument("--threshold", type=float, default=0.5)
    p_davis.add_argument("--seed", type=int, default=0)

    p_dmc = sub.add_parser("dmc", help="visualize rollout on DMC env (no GT)")
    p_dmc.add_argument("--ckpt", type=str, required=True)
    p_dmc.add_argument("--out_dir", type=str, required=True)
    p_dmc.add_argument("--domain_name", type=str, default="walker")
    p_dmc.add_argument("--task_name", type=str, default="walk")
    p_dmc.add_argument("--seed", type=int, default=0)
    p_dmc.add_argument("--episode_length", type=int, default=1000)
    p_dmc.add_argument("--frame_stack", type=int, default=5)
    p_dmc.add_argument("--action_repeat", type=int, default=4)
    p_dmc.add_argument("--image_size", type=int, default=84)
    p_dmc.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "color_easy", "color_hard", "video_easy", "video_hard", "distracting_cs"],
    )
    p_dmc.add_argument("--distracting_cs_intensity", type=float, default=0.1)
    p_dmc.add_argument("--steps", type=int, default=500)
    p_dmc.add_argument("--threshold", type=float, default=0.5)
    p_dmc.add_argument("--apply_rl_aug", action="store_true")
    p_dmc.add_argument("--rl_shift_pad", type=int, default=4)
    p_dmc.add_argument("--rl_overlay_prob", type=float, default=1.0)
    p_dmc.add_argument("--save_every", type=int, default=1)

    args = parser.parse_args()

    if args.cmd == "davis":
        visualize_davis_val(
            davis_root=args.data_root,
            davis_year=args.year,
            resolution=args.resolution,
            sequence_length=int(args.sequence_length),
            target_size=int(args.target_size),
            ckpt_path=args.ckpt,
            out_dir=args.out_dir,
            num_samples=int(args.num_samples),
            threshold=float(args.threshold),
            seed=int(args.seed),
        )
    else:
        visualize_dmc_rollout(
            ckpt_path=args.ckpt,
            out_dir=args.out_dir,
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=int(args.seed),
            episode_length=int(args.episode_length),
            frame_stack=int(args.frame_stack),
            action_repeat=int(args.action_repeat),
            image_size=int(args.image_size),
            env_mode=str(args.mode),
            distracting_cs_intensity=float(args.distracting_cs_intensity),
            steps=int(args.steps),
            threshold=float(args.threshold),
            apply_rl_aug=bool(args.apply_rl_aug),
            rl_shift_pad=int(args.rl_shift_pad),
            rl_overlay_prob=float(args.rl_overlay_prob),
            save_every=int(args.save_every),
        )


if __name__ == "__main__":
    main()
