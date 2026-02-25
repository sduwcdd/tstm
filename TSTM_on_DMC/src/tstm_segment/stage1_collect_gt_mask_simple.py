"""
Stage 1: Directly obtain ground-truth segmentation masks from MuJoCo (simple, DMC integration)

- Create env via env.wrappers.make_env
- Render segmentation using dmc2gym _physics
- Save RGB frames, binary agent masks, and visualizations
- Output dir: logs/temporal_training_data_gt/{domain}_{task}/seed_{seed}
"""
import os
os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import gym
import utils
from tqdm import tqdm
import cv2
from pathlib import Path
import pickle
from env.wrappers import make_env


def get_physics_from_env(env):
    """Downward unwrap env to get dmc2gym's _physics object."""
    _env = env
    while not hasattr(_env, "_physics") and hasattr(_env, "env"):
        _env = _env.env
    if hasattr(_env, "_physics"):
        return _env._physics
    return None


def render_segmentation(env, height, width, camera_id=0):
    """Render segmentation mask.
    Returns:
        seg_image: (H, W, 2) int32 array
            - [:, :, 0]: object ID
            - [:, :, 1]: object type
    """
    physics = get_physics_from_env(env)
    if physics is None:
        raise RuntimeError("Failed to access physics object")
    seg_image = physics.render(
        height=height,
        width=width,
        camera_id=camera_id,
        segmentation=True,
    )
    return seg_image


def extract_agent_mask(seg_image, debug=False):
    """Extract agent mask from segmentation image.
    Strategy: exclude ID=0 (ground/world), all others are considered agent pixels.
    """
    object_ids = seg_image[:, :, 0]
    H, W = object_ids.shape

    unique_ids = np.unique(object_ids)
    unique_ids = unique_ids[unique_ids >= 0]  # Exclude background (-1)

    agent_mask = np.zeros((H, W), dtype=np.uint8)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        agent_mask |= (object_ids == obj_id)
    return agent_mask.astype(np.uint8)


class GTMaskCollector:
    """Ground Truth Mask collector (simple version, DMC integration)"""

    def __init__(self, args, save_dir, max_episodes=100):
        self.save_dir = Path(save_dir)
        self.max_episodes = max_episodes
        self.image_size = args.image_size

        self.frames_dir = self.save_dir / "frames"
        self.masks_dir = self.save_dir / "masks"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 70)
        print("Ground Truth Mask collector (simple version, DMC integration)")
        print("=" * 70)
        print(f"  Resolution: {self.image_size}x{self.image_size}")
        print(f"  Save dir: {self.save_dir}")
        print("=" * 70)

        # Create env (train mode)
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

        # Verify physics access
        physics = get_physics_from_env(self.env)
        if physics is None:
            print("Error: cannot access physics object")
            print(f"  Env type: {type(self.env)}")
            raise RuntimeError("Cannot access physics")
        else:
            print("Physics access OK")

        print("Env ready\n")

    def collect_episodes(self, use_random_policy=True):
        """Collect episodes and GT masks."""
        episode_data = []

        print(f"Collecting {self.max_episodes} episodes of GT masks...")

        for episode_idx in tqdm(range(self.max_episodes), desc="Collecting episodes"):
            obs = self.env.reset()
            done = False
            episode_frames = []
            step_idx = 0

            while not done:
                try:
                    physics = get_physics_from_env(self.env)

                    # Get RGB image
                    rgb_image = physics.render(height=self.image_size, width=self.image_size, camera_id=0, segmentation=False)

                    # Get segmentation (same step)
                    seg_image = physics.render(height=self.image_size, width=self.image_size, camera_id=0, segmentation=True)

                    # Extract agent mask
                    agent_mask = extract_agent_mask(seg_image, debug=False)

                    # Save RGB frame
                    frame_uint8 = rgb_image.astype(np.uint8)
                    frame_path = self.frames_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))

                    # Save mask
                    mask_path = self.masks_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.npy"
                    np.save(mask_path, agent_mask)

                    # Save visualization
                    vis_path = self.masks_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}_vis.png"
                    self.save_visualization(frame_uint8, agent_mask, vis_path)

                    episode_frames.append({
                        'frame_path': str(frame_path),
                        'mask_path': str(mask_path),
                        'step': step_idx,
                        'action': None
                    })

                except Exception as e:
                    print(f"\nWarning: episode {episode_idx}, step {step_idx}: {e}")

                # Step action
                action = self.env.action_space.sample() if use_random_policy else self.env.action_space.sample()
                obs, reward, done, info = self.env.step(action)
                step_idx += 1

            episode_data.append({
                'episode_idx': episode_idx,
                'num_steps': step_idx,
                'frames': episode_frames,
            })

        # Save metadata
        metadata_path = self.save_dir / "episodes_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(episode_data, f)

        total_frames = sum(len(ep['frames']) for ep in episode_data)
        print(f"\nCollection done!")
        print(f"  Total frames: {total_frames}")
        print(f"  Metadata: {metadata_path}")

        return episode_data

    def save_visualization(self, image, mask, save_path):
        """Save mask visualization image."""
        overlay = image.copy()

        # Draw green contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Alpha blend fill
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 255]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        result = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), result)

        # Save raw mask
        mask_bw_path = str(save_path).replace('_vis.png', '_mask_bw.png')
        cv2.imwrite(mask_bw_path, mask * 255)


def main(args):
    print("\n" + "=" * 70)
    print("Stage 1: Collect ground-truth masks (DMC integration)")
    print("=" * 70)
    print(f"Env: {args.domain_name}_{args.task_name}")
    print(f"Episodes to collect: {args.collect_episodes}")
    print(f"Seed: {args.seed}")
    print("=" * 70 + "\n")

    utils.set_seed_everywhere(args.seed)

    # Create save directory (relative to project root)
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.log_dir / "temporal_training_data_gt" / f"{args.domain_name}_{args.task_name}" / f"seed_{args.seed}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    collector = GTMaskCollector(args, data_dir, max_episodes=args.collect_episodes)
    _ = collector.collect_episodes(use_random_policy=True)

    print("\n" + "=" * 70)
    print(f"Done! Data saved to: {data_dir}")
    print("=" * 70)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect ground-truth masks (DMC integration)')

    parser.add_argument('--domain_name', default='walker')
    parser.add_argument('--task_name', default='walk')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frame_stack', default=8, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)
    parser.add_argument('--episode_length', default=1000, type=int)
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--eval_mode', default='all', type=str)
    parser.add_argument('--distracting_cs_intensity', default=0.1, type=float)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--collect_episodes', type=int, default=100)

    args = parser.parse_args()
    args.image_crop_size = args.image_size
    args.encoder = 'cnn'
    args.algorithm = 'sac'

    main(args)
