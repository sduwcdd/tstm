"""
Stage 1 (pseudo labels): Collect RGB frames from DMC and generate pseudo segmentation masks
using optical flow (no ground-truth masks).

- Create env via env.wrappers.make_env
- Render consecutive RGB frames from dmc2gym _physics
- Compute dense optical flow (Farneback) and threshold motion magnitude to obtain a binary mask
- Save RGB frames, binary pseudo masks, and visualizations
- Output dir: logs/temporal_training_data_flow/{domain}_{task}/seed_{seed}
"""

import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import sys

import pickle
from pathlib import Path

import cv2
import gym
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import utils
from env.wrappers import make_env


def get_physics_from_env(env):
    """Downward unwrap env to get dmc2gym's _physics object."""
    _env = env
    while not hasattr(_env, "_physics") and hasattr(_env, "env"):
        _env = _env.env
    if hasattr(_env, "_physics"):
        return _env._physics
    return None


def _to_gray01(rgb_uint8: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def _filter_small_components(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask_u8

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    out = np.zeros_like(mask_u8)
    for lab in range(1, num_labels):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == lab] = 1
    return out


def _ellipse_kernel(k: int) -> np.ndarray:
    kk = int(k) if k is not None else 3
    if kk < 2:
        kk = 3
    if kk % 2 == 0:
        kk += 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))


def _keep_components_by_area(mask_u8: np.ndarray, rel_area: float, max_components: int) -> np.ndarray:
    if mask_u8.max() == 0:
        return mask_u8
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8

    areas = []
    for lab in range(1, num_labels):
        areas.append((lab, int(stats[lab, cv2.CC_STAT_AREA])))
    areas.sort(key=lambda x: x[1], reverse=True)
    if not areas:
        return mask_u8

    largest_area = float(areas[0][1])
    thr = float(rel_area) * largest_area
    k = int(max_components) if max_components is not None else len(areas)
    k = max(1, k)

    out = np.zeros_like(mask_u8)
    kept = 0
    for lab, area in areas:
        if kept >= k:
            break
        if float(area) >= thr:
            out[labels == lab] = 1
            kept += 1
    return out


def _erode_to_coverage(mask_u8: np.ndarray, coverage_max: float, kernel_size: int, max_iters: int) -> np.ndarray:
    if mask_u8.max() == 0:
        return mask_u8
    if coverage_max is None:
        return mask_u8
    cmax = float(coverage_max)
    if cmax <= 0.0:
        return np.zeros_like(mask_u8)
    if float(mask_u8.mean()) <= cmax:
        return mask_u8

    kernel = _ellipse_kernel(kernel_size if kernel_size is not None else 3)
    iters = int(max_iters) if max_iters is not None else 0
    iters = max(0, iters)

    out = mask_u8
    for _ in range(iters):
        if float(out.mean()) <= cmax:
            break
        out = cv2.erode(out, kernel, iterations=1)
        if out.max() == 0:
            break
    return out


def _grabcut_refine_adaptive_bbox(
    curr_rgb_uint8: np.ndarray,
    seed: np.ndarray,
    bbox_pad: int,
    grabcut_iters: int,
    grabcut_fg_erode_kernel: int,
    grabcut_fg_erode_iters: int,
    grabcut_pr_dilate_kernel: int,
    grabcut_pr_dilate_iters: int,
) -> np.ndarray:
    img_bgr = cv2.cvtColor(curr_rgb_uint8, cv2.COLOR_RGB2BGR)

    ys, xs = np.where(seed > 0)
    sy0, sy1 = int(ys.min()), int(ys.max())
    sx0, sx1 = int(xs.min()), int(xs.max())
    base_pad = int(bbox_pad) if bbox_pad is not None else 0

    fg_seed = seed.copy()
    if (
        grabcut_fg_erode_kernel is not None
        and int(grabcut_fg_erode_kernel) > 1
        and grabcut_fg_erode_iters is not None
        and int(grabcut_fg_erode_iters) > 0
    ):
        kernel = _ellipse_kernel(int(grabcut_fg_erode_kernel))
        fg_seed = cv2.erode(fg_seed, kernel, iterations=int(grabcut_fg_erode_iters))

    pr_fg = seed.copy()
    if (
        grabcut_pr_dilate_kernel is not None
        and int(grabcut_pr_dilate_kernel) > 1
        and grabcut_pr_dilate_iters is not None
        and int(grabcut_pr_dilate_iters) > 0
    ):
        kernel = _ellipse_kernel(int(grabcut_pr_dilate_kernel))
        pr_fg = cv2.dilate(pr_fg, kernel, iterations=int(grabcut_pr_dilate_iters))

    iters = int(grabcut_iters) if grabcut_iters is not None else 1
    iters = max(1, iters)

    def _run(pad: int):
        y0 = max(0, sy0 - pad)
        x0 = max(0, sx0 - pad)
        y1 = min(seed.shape[0] - 1, sy1 + pad)
        x1 = min(seed.shape[1] - 1, sx1 + pad)

        gc_mask = np.full(seed.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        gc_mask[pr_fg > 0] = cv2.GC_PR_FGD
        gc_mask[fg_seed > 0] = cv2.GC_FGD
        gc_mask[:y0, :] = cv2.GC_BGD
        gc_mask[y1 + 1 :, :] = cv2.GC_BGD
        gc_mask[:, :x0] = cv2.GC_BGD
        gc_mask[:, x1 + 1 :] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        cv2.grabCut(img_bgr, gc_mask, None, bgdModel, fgdModel, iters, mode=cv2.GC_INIT_WITH_MASK)
        out = np.where((gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 1, 0).astype(np.uint8)

        touch = False
        if out.max() > 0:
            if y0 > 0 and out[y0, x0 : x1 + 1].max() > 0:
                touch = True
            if y1 < out.shape[0] - 1 and out[y1, x0 : x1 + 1].max() > 0:
                touch = True
            if x0 > 0 and out[y0 : y1 + 1, x0].max() > 0:
                touch = True
            if x1 < out.shape[1] - 1 and out[y0 : y1 + 1, x1].max() > 0:
                touch = True

        return out, touch

    mask0, touch0 = _run(base_pad)
    if touch0:
        pad2 = base_pad + max(8, base_pad // 2)
        mask1, _ = _run(pad2)
        return mask1 if mask1.max() > 0 else mask0

    return mask0


def compute_flow_pseudo_mask(
    prev_rgb_uint8: np.ndarray,
    curr_rgb_uint8: np.ndarray,
    motion_thr: float,
    motion_quantile: float,
    coverage_min: float,
    coverage_max: float,
    final_coverage_max: float,
    coverage_adjust_iters: int,
    coverage_adjust_step: float,
    flow_weight: float,
    diff_weight: float,
    score_blur: float,
    use_grabcut: int,
    grabcut_iters: int,
    grabcut_fg_erode_kernel: int,
    grabcut_fg_erode_iters: int,
    grabcut_pr_dilate_kernel: int,
    grabcut_pr_dilate_iters: int,
    final_erode_kernel: int,
    final_erode_iters_max: int,
    morph_kernel: int,
    open_iters: int,
    close_iters: int,
    dilate_kernel: int,
    dilate_iters: int,
    keep_largest: int,
    keep_rel_area: float,
    keep_max_components: int,
    bbox_fill: int,
    bbox_pad: int,
    min_area: int,
    motion_gate_quantile: float,
    motion_gate_dilate_kernel: int,
    motion_gate_dilate_iters: int,
):
    prev_gray = _to_gray01(prev_rgb_uint8)
    curr_gray = _to_gray01(curr_rgb_uint8)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    diff = np.abs(curr_gray - prev_gray)

    score = float(flow_weight) * mag + float(diff_weight) * diff

    if score_blur is not None and float(score_blur) > 0.0:
        sigma = float(score_blur)
        score = cv2.GaussianBlur(score.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma)

    # 1) Seed mask by thresholding motion score (cheap coverage control happens here)
    if motion_thr is not None and float(motion_thr) > 0.0:
        thr = float(motion_thr)
        seed = (score > thr).astype(np.uint8)
    else:
        q = float(motion_quantile)
        q = min(max(q, 0.0), 1.0)
        cmin = float(coverage_min) if coverage_min is not None else 0.0
        cmax = float(coverage_max) if coverage_max is not None else 1.0
        cmin = min(max(cmin, 0.0), 1.0)
        cmax = min(max(cmax, 0.0), 1.0)
        if cmax < cmin:
            cmax = cmin

        max_it = int(coverage_adjust_iters) if coverage_adjust_iters is not None else 0
        step = float(coverage_adjust_step) if coverage_adjust_step is not None else 0.0
        max_it = max(0, max_it)
        step = max(0.0, step)

        thr = float(np.quantile(score, q))
        seed = (score > thr).astype(np.uint8)
        if max_it > 0 and step > 0.0:
            for _ in range(max_it):
                cov = float(seed.mean())
                if cov > cmax and q < 0.995:
                    q = min(0.995, q + step)
                elif cov < cmin and q > 0.01:
                    q = max(0.01, q - step)
                else:
                    break
                thr = float(np.quantile(score, q))
                seed = (score > thr).astype(np.uint8)

    if seed.max() == 0:
        return np.zeros_like(curr_gray, dtype=np.uint8)

    seed = _filter_small_components(seed, int(min_area))
    if seed.max() == 0:
        return np.zeros_like(curr_gray, dtype=np.uint8)

    # 2) GrabCut refinement (expand from seeds, but keep conservative FG init)
    mask = seed
    grabcut_enabled = use_grabcut is not None and int(use_grabcut) != 0
    if grabcut_enabled:
        try:
            mask = _grabcut_refine_adaptive_bbox(
                curr_rgb_uint8=curr_rgb_uint8,
                seed=seed,
                bbox_pad=bbox_pad,
                grabcut_iters=grabcut_iters,
                grabcut_fg_erode_kernel=grabcut_fg_erode_kernel,
                grabcut_fg_erode_iters=grabcut_fg_erode_iters,
                grabcut_pr_dilate_kernel=grabcut_pr_dilate_kernel,
                grabcut_pr_dilate_iters=grabcut_pr_dilate_iters,
            )
        except Exception:
            mask = seed

    # 3) Post-processing for stability
    if dilate_kernel is not None and int(dilate_kernel) > 1 and dilate_iters is not None and int(dilate_iters) > 0:
        kernel = _ellipse_kernel(int(dilate_kernel))
        mask = cv2.dilate(mask, kernel, iterations=int(dilate_iters))

    if morph_kernel is not None and morph_kernel > 1:
        kernel = _ellipse_kernel(int(morph_kernel))
        if open_iters is not None and open_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=int(open_iters))
        if close_iters is not None and close_iters > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))

    if motion_gate_quantile is not None and float(motion_gate_quantile) > 0.0:
        gq = float(motion_gate_quantile)
        gq = min(max(gq, 0.0), 1.0)
        gthr = float(np.quantile(score, gq))
        gate = (score > gthr).astype(np.uint8)
        if motion_gate_dilate_kernel is not None and int(motion_gate_dilate_kernel) > 1 and motion_gate_dilate_iters is not None and int(motion_gate_dilate_iters) > 0:
            kernel = _ellipse_kernel(int(motion_gate_dilate_kernel))
            gate = cv2.dilate(gate, kernel, iterations=int(motion_gate_dilate_iters))
        gated = (mask & gate).astype(np.uint8)
        if gated.max() > 0:
            mask = gated

    mask = _filter_small_components(mask, int(min_area))
    if keep_largest is not None and int(keep_largest) != 0:
        rel = float(keep_rel_area) if keep_rel_area is not None else 0.1
        rel = max(0.0, rel)
        mask = _keep_components_by_area(mask, rel_area=rel, max_components=keep_max_components)

    if morph_kernel is not None and morph_kernel > 1 and close_iters is not None and int(close_iters) > 0:
        kernel = _ellipse_kernel(int(morph_kernel))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Final: if mask is too large, erode it down to coverage_max (prevents background swallowing)
    cmax_final = final_coverage_max if final_coverage_max is not None else coverage_max
    mask = _erode_to_coverage(mask, cmax_final, final_erode_kernel, final_erode_iters_max)

    if bbox_fill is not None and int(bbox_fill) != 0 and mask.max() > 0:
        ys, xs = np.where(mask > 0)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        pad = int(bbox_pad) if bbox_pad is not None else 0
        y0 = max(0, y0 - pad)
        x0 = max(0, x0 - pad)
        y1 = min(mask.shape[0] - 1, y1 + pad)
        x1 = min(mask.shape[1] - 1, x1 + pad)
        filled = np.zeros_like(mask)
        filled[y0 : y1 + 1, x0 : x1 + 1] = 1
        mask = filled

    return mask


class FlowPseudoMaskCollector:
    def __init__(self, args, save_dir, max_episodes=100):
        self.save_dir = Path(save_dir)
        self.max_episodes = max_episodes
        self.image_size = args.image_size

        self.frames_dir = self.save_dir / "frames"
        self.masks_dir = self.save_dir / "masks"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self.motion_thr = args.motion_thr
        self.motion_quantile = args.motion_quantile
        self.coverage_min = args.coverage_min
        self.coverage_max = args.coverage_max
        self.final_coverage_max = args.final_coverage_max
        self.coverage_adjust_iters = args.coverage_adjust_iters
        self.coverage_adjust_step = args.coverage_adjust_step
        self.flow_weight = args.flow_weight
        self.diff_weight = args.diff_weight
        self.score_blur = args.score_blur
        self.use_grabcut = args.use_grabcut
        self.grabcut_iters = args.grabcut_iters
        self.grabcut_fg_erode_kernel = args.grabcut_fg_erode_kernel
        self.grabcut_fg_erode_iters = args.grabcut_fg_erode_iters
        self.grabcut_pr_dilate_kernel = args.grabcut_pr_dilate_kernel
        self.grabcut_pr_dilate_iters = args.grabcut_pr_dilate_iters
        self.final_erode_kernel = args.final_erode_kernel
        self.final_erode_iters_max = args.final_erode_iters_max
        self.morph_kernel = args.morph_kernel
        self.open_iters = args.open_iters
        self.close_iters = args.close_iters
        self.dilate_kernel = args.dilate_kernel
        self.dilate_iters = args.dilate_iters
        self.keep_largest = args.keep_largest
        self.keep_rel_area = args.keep_rel_area
        self.keep_max_components = args.keep_max_components
        self.bbox_fill = args.bbox_fill
        self.bbox_pad = args.bbox_pad
        self.min_area = args.min_area
        self.motion_gate_quantile = args.motion_gate_quantile
        self.motion_gate_dilate_kernel = args.motion_gate_dilate_kernel
        self.motion_gate_dilate_iters = args.motion_gate_dilate_iters

        print("=" * 70)
        print("Pseudo Mask collector (optical flow)")
        print("=" * 70)
        print(f"  Resolution: {self.image_size}x{self.image_size}")
        print(f"  Save dir: {self.save_dir}")
        print(f"  motion_thr: {self.motion_thr}")
        print(f"  motion_quantile: {self.motion_quantile}")
        print(
            f"  coverage_min/max: {self.coverage_min}/{self.coverage_max} (final_max: {self.final_coverage_max}, adjust iters/step: {self.coverage_adjust_iters}/{self.coverage_adjust_step})"
        )
        print(f"  flow_weight: {self.flow_weight}, diff_weight: {self.diff_weight}")
        print(f"  score_blur: {self.score_blur}")
        print(f"  use_grabcut: {self.use_grabcut}, grabcut_iters: {self.grabcut_iters}")
        print(
            "  grabcut seeds(erode k/it, pr-dilate k/it): "
            f"{self.grabcut_fg_erode_kernel}/{self.grabcut_fg_erode_iters}, {self.grabcut_pr_dilate_kernel}/{self.grabcut_pr_dilate_iters}"
        )
        print(f"  final_erode(k/iters_max): {self.final_erode_kernel}/{self.final_erode_iters_max}")
        print(
            f"  morph_kernel: {self.morph_kernel}, open_iters: {self.open_iters}, close_iters: {self.close_iters}"
        )
        print(
            f"  dilate_kernel: {self.dilate_kernel}, dilate_iters: {self.dilate_iters}, keep_largest: {self.keep_largest}"
        )
        print(f"  bbox_fill: {self.bbox_fill}, bbox_pad: {self.bbox_pad}")
        print(f"  min_area: {self.min_area}")
        print(
            f"  motion_gate(q/k/it): {self.motion_gate_quantile}/{self.motion_gate_dilate_kernel}/{self.motion_gate_dilate_iters}"
        )
        print("=" * 70)

        print("Creating env...")
        gym.logger.set_level(40)
        self.env = make_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed,
            episode_length=args.episode_length,
            frame_stack=args.frame_stack,
            action_repeat=args.action_repeat,
            image_size=args.image_size,
            mode="train",
            intensity=args.distracting_cs_intensity,
        )

        physics = get_physics_from_env(self.env)
        if physics is None:
            print("Error: cannot access physics object")
            print(f"  Env type: {type(self.env)}")
            raise RuntimeError("Cannot access physics")
        print("Physics access OK")
        print("Env ready\n")

    def collect_episodes(self, use_random_policy=True):
        episode_data = []

        print(f"Collecting {self.max_episodes} episodes of pseudo masks...")

        from tqdm import tqdm

        coverages = []

        for episode_idx in tqdm(range(self.max_episodes), desc="Collecting episodes"):
            _ = self.env.reset()
            done = False
            episode_frames = []
            step_idx = 0

            prev_rgb = None

            while not done:
                try:
                    physics = get_physics_from_env(self.env)
                    rgb_image = physics.render(
                        height=self.image_size,
                        width=self.image_size,
                        camera_id=0,
                        segmentation=False,
                    )

                    frame_uint8 = rgb_image.astype(np.uint8)

                    if prev_rgb is None:
                        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
                    else:
                        mask = compute_flow_pseudo_mask(
                            prev_rgb_uint8=prev_rgb,
                            curr_rgb_uint8=frame_uint8,
                            motion_thr=self.motion_thr,
                            motion_quantile=self.motion_quantile,
                            coverage_min=self.coverage_min,
                            coverage_max=self.coverage_max,
                            final_coverage_max=self.final_coverage_max,
                            coverage_adjust_iters=self.coverage_adjust_iters,
                            coverage_adjust_step=self.coverage_adjust_step,
                            flow_weight=self.flow_weight,
                            diff_weight=self.diff_weight,
                            score_blur=self.score_blur,
                            use_grabcut=self.use_grabcut,
                            grabcut_iters=self.grabcut_iters,
                            grabcut_fg_erode_kernel=self.grabcut_fg_erode_kernel,
                            grabcut_fg_erode_iters=self.grabcut_fg_erode_iters,
                            grabcut_pr_dilate_kernel=self.grabcut_pr_dilate_kernel,
                            grabcut_pr_dilate_iters=self.grabcut_pr_dilate_iters,
                            final_erode_kernel=self.final_erode_kernel,
                            final_erode_iters_max=self.final_erode_iters_max,
                            morph_kernel=self.morph_kernel,
                            open_iters=self.open_iters,
                            close_iters=self.close_iters,
                            dilate_kernel=self.dilate_kernel,
                            dilate_iters=self.dilate_iters,
                            keep_largest=self.keep_largest,
                            keep_rel_area=self.keep_rel_area,
                            keep_max_components=self.keep_max_components,
                            bbox_fill=self.bbox_fill,
                            bbox_pad=self.bbox_pad,
                            min_area=self.min_area,
                            motion_gate_quantile=self.motion_gate_quantile,
                            motion_gate_dilate_kernel=self.motion_gate_dilate_kernel,
                            motion_gate_dilate_iters=self.motion_gate_dilate_iters,
                        )

                    coverages.append(float(mask.mean()))

                    frame_path = self.frames_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

                    mask_path = self.masks_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.npy"
                    np.save(mask_path, mask.astype(np.uint8))

                    vis_path = self.masks_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}_vis.png"
                    self.save_visualization(frame_uint8, mask, vis_path)

                    episode_frames.append(
                        {
                            "frame_path": str(frame_path),
                            "mask_path": str(mask_path),
                            "step": step_idx,
                            "action": None,
                        }
                    )

                    prev_rgb = frame_uint8

                except Exception as e:
                    print(f"\nWarning: episode {episode_idx}, step {step_idx}: {e}")

                action = self.env.action_space.sample()
                _, _, done, _ = self.env.step(action)
                step_idx += 1

            episode_data.append({"episode_idx": episode_idx, "num_steps": step_idx, "frames": episode_frames})

        metadata_path = self.save_dir / "episodes_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(episode_data, f)

        total_frames = sum(len(ep["frames"]) for ep in episode_data)
        print("\nCollection done!")
        print(f"  Total frames: {total_frames}")
        print(f"  Metadata: {metadata_path}")
        if coverages:
            c = np.asarray(coverages, dtype=np.float32)
            print(
                "  mask_coverage(mean/p50/p90): "
                f"{float(c.mean()):.4f} / {float(np.quantile(c, 0.5)):.4f} / {float(np.quantile(c, 0.9)):.4f}"
            )

        return episode_data

    def save_visualization(self, image, mask, save_path):
        overlay = image.copy()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 255]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        result = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), result)

        mask_bw_path = str(save_path).replace("_vis.png", "_mask_bw.png")
        cv2.imwrite(mask_bw_path, mask.astype(np.uint8) * 255)


def main(args):
    print("\n" + "=" * 70)
    print("Stage 1 (pseudo): Collect optical-flow pseudo masks")
    print("=" * 70)
    print(f"Env: {args.domain_name}_{args.task_name}")
    print(f"Episodes to collect: {args.collect_episodes}")
    print(f"Seed: {args.seed}")
    print("=" * 70 + "\n")

    utils.set_seed_everywhere(args.seed)

    project_root = Path(__file__).resolve().parents[2]
    data_dir = (
        project_root
        / args.log_dir
        / "temporal_training_data_flow"
        / f"{args.domain_name}_{args.task_name}"
        / f"seed_{args.seed}"
    )
    data_dir.mkdir(parents=True, exist_ok=True)

    collector = FlowPseudoMaskCollector(args, data_dir, max_episodes=args.collect_episodes)
    _ = collector.collect_episodes(use_random_policy=True)

    print("\n" + "=" * 70)
    print(f"Done! Data saved to: {data_dir}")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect optical-flow pseudo masks (DMC integration)")

    parser.add_argument("--domain_name", default="walker")
    parser.add_argument("--task_name", default="walk")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame_stack", default=8, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--distracting_cs_intensity", default=0.1, type=float)
    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--collect_episodes", type=int, default=100)

    parser.add_argument("--motion_thr", type=float, default=0.0)
    parser.add_argument("--motion_quantile", type=float, default=0.90)
    parser.add_argument("--coverage_min", type=float, default=0.02)
    parser.add_argument("--coverage_max", type=float, default=0.35)
    parser.add_argument("--final_coverage_max", type=float, default=0.30)
    parser.add_argument("--coverage_adjust_iters", type=int, default=10)
    parser.add_argument("--coverage_adjust_step", type=float, default=0.02)
    parser.add_argument("--flow_weight", type=float, default=1.0)
    parser.add_argument("--diff_weight", type=float, default=0.5)

    parser.add_argument("--score_blur", type=float, default=1.0)

    parser.add_argument("--use_grabcut", type=int, default=1)
    parser.add_argument("--grabcut_iters", type=int, default=2)
    parser.add_argument("--grabcut_fg_erode_kernel", type=int, default=5)
    parser.add_argument("--grabcut_fg_erode_iters", type=int, default=0)
    parser.add_argument("--grabcut_pr_dilate_kernel", type=int, default=5)
    parser.add_argument("--grabcut_pr_dilate_iters", type=int, default=2)

    parser.add_argument("--final_erode_kernel", type=int, default=5)
    parser.add_argument("--final_erode_iters_max", type=int, default=2)

    parser.add_argument("--morph_kernel", type=int, default=7)
    parser.add_argument("--open_iters", type=int, default=0)
    parser.add_argument("--close_iters", type=int, default=2)

    parser.add_argument("--dilate_kernel", type=int, default=7)
    parser.add_argument("--dilate_iters", type=int, default=1)
    parser.add_argument("--keep_largest", type=int, default=1)
    parser.add_argument("--keep_rel_area", type=float, default=0.03)
    parser.add_argument("--keep_max_components", type=int, default=10)
    parser.add_argument("--bbox_fill", type=int, default=0)
    parser.add_argument("--bbox_pad", type=int, default=8)

    parser.add_argument("--min_area", type=int, default=5)

    parser.add_argument("--motion_gate_quantile", type=float, default=0.0)
    parser.add_argument("--motion_gate_dilate_kernel", type=int, default=9)
    parser.add_argument("--motion_gate_dilate_iters", type=int, default=2)

    args = parser.parse_args()
    args.image_crop_size = args.image_size
    args.encoder = "cnn"
    args.algorithm = "sac"

    main(args)
