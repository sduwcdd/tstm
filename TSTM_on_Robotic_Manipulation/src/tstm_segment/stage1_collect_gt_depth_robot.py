"""
Stage 1: Collect ground-truth masks using depth-based segmentation (robot_env)

- Use `generate_full_scene_mask` to get binary masks
- Save RGB frames, binary masks and visualizations
- Output dir: logs/vos_training_data_gt/{domain}_{task}/seed_{seed}
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
from env.robot.segmentation_utils import generate_full_scene_mask
from env.robot.segmentation_config import get_segmentation_config


def get_sim_from_env(env):
    """Unwrap env to get MuJoCo sim object."""
    _env = env
    while not hasattr(_env, "sim") and hasattr(_env, "env"):
        _env = _env.env
    if hasattr(_env, "sim"):
        return _env.sim
    return None


def get_camera_name_from_env(env):
    """Get the active camera name from env."""
    _env = env
    while not hasattr(_env, "cameras") and hasattr(_env, "env"):
        _env = _env.env
    if hasattr(_env, "cameras") and len(_env.cameras) > 0:
        return _env.cameras[0]
    return "third_person"


class GTMaskCollectorDepth:
    """Ground-truth mask collector (depth-based, robot_env)."""

    def __init__(self, args, save_dir, max_episodes=50):
        self.save_dir = Path(save_dir)
        self.max_episodes = max_episodes
        self.image_size = args.image_size
        self.task_name = args.task_name

        self.frames_dir = self.save_dir / "frames"
        self.masks_dir = self.save_dir / "masks"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)


        # Get task
        self.seg_config = get_segmentation_config(args.task_name)

        # Create env
        gym.logger.set_level(40)
        self.env = make_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed,
            episode_length=args.episode_length,
            n_substeps=args.n_substeps,
            frame_stack=args.frame_stack,
            image_size=args.image_size,
            cameras=[args.camera],
            mode="train",
            observation_type="image",
            action_space=args.action_space,
        )

        # Verify sim access
        self.sim = get_sim_from_env(self.env)
        if self.sim is None:
            print("Error: cannot access MuJoCo sim")
            raise RuntimeError("Cannot access sim")
        else:
            pass

        self.camera_name = get_camera_name_from_env(self.env)
        

    def collect_episodes(self, use_random_policy=True):
        """Collect episodes and GT masks (depth-based)."""
        episode_data = []

        

        for episode_idx in tqdm(range(self.max_episodes), desc="Collecting episodes"):
            obs, state_obs = self.env.reset()
            done = False
            episode_frames = []
            step_idx = 0

            while not done:
                try:
                    # Generate a mask using depth map segmentation
                    mask = generate_full_scene_mask(
                        self.sim,
                        has_object=self.seg_config['has_object'],
                        object_site=self.seg_config['object_site'],
                        include_target=self.seg_config['include_target'],
                        target_site=self.seg_config['target_site'],
                        camera_name=self.camera_name,
                        width=self.image_size,
                        height=self.image_size,
                    )
                    obs_array = np.array(obs)
                    if len(obs_array.shape) == 3:
                        rgb_image = obs_array.transpose(1, 2, 0)  # -> (H, W, C)
                    else:
                        rgb_image = obs_array[-3:].transpose(1, 2, 0)  # -> (H, W, 3)
                    if rgb_image.max() <= 1.0:
                        rgb_image = (rgb_image * 255).astype(np.uint8)
                    else:
                        rgb_image = rgb_image.astype(np.uint8)
                        
                    frame_path = self.frames_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.png"
                    cv2.imwrite(str(frame_path), cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

                    mask_path = self.masks_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}.npy"
                    np.save(mask_path, mask.astype(np.uint8))

                    vis_path = self.masks_dir / f"episode_{episode_idx:04d}_step_{step_idx:04d}_vis.png"
                    self.save_visualization(rgb_image, mask, vis_path)

                    episode_frames.append({
                        'frame_path': str(frame_path),
                        'mask_path': str(mask_path),
                        'step': step_idx,
                        'action': None
                    })

                except Exception as e:
                    print(f"\nWarning: episode {episode_idx}, step {step_idx}: {e}")

                action = self.env.action_space.sample()
                obs, state_obs, reward, done, info = self.env.step(action)
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
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)

        # Alpha blend fill (cyan)
        mask_colored = np.zeros_like(image)
        mask_colored[mask > 0] = [0, 255, 255]
        overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)

        result = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(save_path), result)

        # Save raw mask
        mask_bw_path = str(save_path).replace('_vis.png', '_mask_bw.png')
        cv2.imwrite(mask_bw_path, mask.astype(np.uint8) * 255)


def main(args):
    

    utils.set_seed_everywhere(args.seed)

    # Create save directory
    project_root = Path(__file__).resolve().parents[2]
    data_dir = project_root / args.log_dir / "vos_training_data_gt" / f"robot_{args.task_name}" / f"seed_{args.seed}"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Collect data
    collector = GTMaskCollectorDepth(args, data_dir, max_episodes=args.collect_episodes)
    _ = collector.collect_episodes(use_random_policy=True)

    print(f"Done. Data: {data_dir}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Collect GT masks via depth-based segmentation (robot_env)')

    # Env params
    parser.add_argument('--domain_name', default='robot', type=str)
    parser.add_argument('--task_name', default='reach', type=str, help='Task name: reach, push, pegbox, hammerall')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--frame_stack', default=1, type=int)
    parser.add_argument('--episode_length', default=50, type=int)
    parser.add_argument('--n_substeps', default=20, type=int)
    parser.add_argument('--image_size', type=int, default=84)
    parser.add_argument('--camera', default='third_person', type=str, help='Camera name: third_person, first_person')
    parser.add_argument('--action_space', default='xy', type=str, help='Action space: xy (reach/push), xyz (pegbox/hammerall)')
    
    # Data collection params
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--collect_episodes', type=int, default=50, help='Number of episodes to collect (robot tasks are short; 50 is enough)')

    args = parser.parse_args()

    main(args)
