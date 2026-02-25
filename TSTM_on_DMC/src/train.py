import os
os.environ['OMP_NUM_THREADS'] = '1'#Reduce the CPU usage of individual programmes without noticeably slowing them down.

# Must be set before importing dm_control (imported transitively via env.wrappers -> dmc2gym).
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import numpy as np
import gym
# from algorithms.rl_utils import make_obs_grad_grid
import utils
import augmentations
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder


def _get_physics_from_env(env):
    _env = env
    while not hasattr(_env, "_physics") and hasattr(_env, "env"):
        _env = _env.env
    if hasattr(_env, "_physics"):
        return _env._physics
    if hasattr(_env, "physics"):
        return _env.physics
    return None


def _extract_agent_mask(seg_image):
    object_ids = seg_image[:, :, 0]
    unique_ids = np.unique(object_ids)
    unique_ids = unique_ids[unique_ids >= 0]
    agent_mask = np.zeros_like(object_ids, dtype=np.uint8)
    for obj_id in unique_ids:
        if obj_id == 0:
            continue
        agent_mask |= (object_ids == obj_id)
    return agent_mask.astype(np.uint8)


def evaluate(env, agent,algorithm, video, num_episodes, L, step, test_env=False, eval_mode=None):
    episode_rewards = []
    total_steps = 0
    for i in range(num_episodes):
        obs = env.reset()
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        episode_step = 0
        torch_obs = []
        torch_action = []
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)

                obs, reward, done, _ = env.step(action)
                video.record(env, mode=eval_mode)
                episode_reward += reward
                # log in tensorboard 15th step
                # if algorithm == 'sgsac':
                #     if i == 0 and episode_step in [15, 16, 17, 18] and step > 0:
                #         _obs = agent._obs_to_input(obs)
                #         torch_obs.append(_obs)
                #         torch_action.append(
                #             torch.tensor(action).to(_obs.device).unsqueeze(0)
                #         )
                #         prefix = "eval" if eval_mode is None else eval_mode
                #     if i == 0 and episode_step == 18 and step > 0:
                #         agent.log_tensorboard(
                #             torch.cat(torch_obs, 0),
                #             torch.cat(torch_action, 0),
                #             step,
                #             prefix=prefix,
                #         )
                    # attrib_grid = make_obs_grad_grid(torch.sigmoid(mask))
                    # agent.writer.add_image(
                    #     prefix + "/smooth_attrib", attrib_grid, global_step=step
                    # )

                episode_step += 1
                total_steps += 1

        if L is not None:
            _test_env = f"_test_env_{eval_mode}" if test_env else ""
            video.save(f"{step}{_test_env}.mp4")
            L.log(f"eval/episode_reward{_test_env}", episode_reward, step)
        episode_rewards.append(episode_reward)

    return np.mean(episode_rewards), total_steps


def main(args):
    # MuJoCo 2.1.0+ does not require license keys
    # home = os.environ["HOME"]
    # os.environ["MJKEY_PATH"] = f"{home}/.mujoco/mujoco200_linux/bin/mjkey.txt"
    os.environ.setdefault("MUJOCO_GL", "egl")
    # Set seed
    utils.set_seed_everywhere(args.seed)

    # Initialize environments
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        mode="train",
    )
    test_envs = None
    test_envs_mode = None
    if args.eval_mode is not None:
        test_envs = []
        test_envs_mode = []

    if args.eval_mode in [
        "color_easy",
        "color_hard",
        "video_easy",
        "video_hard",
        "distracting_cs",
    ]:

        test_env = make_env(
            domain_name=args.domain_name,
            task_name=args.task_name,
            seed=args.seed + 42,
            episode_length=args.episode_length,
            action_repeat=args.action_repeat,
            image_size=args.image_size,
            frame_stack=args.frame_stack,
            mode=args.eval_mode,
            intensity=args.distracting_cs_intensity,
        )

        test_envs.append(test_env)
        test_envs_mode.append(args.eval_mode)

    if args.eval_mode == "all":
        for eval_mode in ["color_hard", "video_easy", "video_hard"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                frame_stack=args.frame_stack,
                mode=eval_mode,
                intensity=args.distracting_cs_intensity,
            )
            test_envs.append(test_env)
            test_envs_mode.append(eval_mode)
            
    if args.eval_mode == "both":
        for eval_mode in ["video_easy", "video_hard"]:
            test_env = make_env(
                domain_name=args.domain_name,
                task_name=args.task_name,
                seed=args.seed + 42,
                episode_length=args.episode_length,
                action_repeat=args.action_repeat,
                image_size=args.image_size,
                frame_stack=args.frame_stack,
                mode=eval_mode,
                intensity=args.distracting_cs_intensity,
            )
            test_envs.append(test_env)
            test_envs_mode.append(eval_mode)
            
            
    # Create working directory
    work_dir = os.path.join(
        args.log_dir,
        args.domain_name + "_" + args.task_name,
        args.algorithm,
        str(args.seed),
    )
    print("Working directory:", work_dir)
    assert not os.path.exists(
        os.path.join(work_dir, "train.log")
    ), "specified working directory already exists"
    utils.make_dir(work_dir)
    model_dir = utils.make_dir(os.path.join(work_dir, "model"))
    video_dir = utils.make_dir(os.path.join(work_dir, "video"))
    video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)
    utils.write_info(args, os.path.join(work_dir, "info.log"))

    # Prepare agent
    assert torch.cuda.is_available(), "must have cuda enabled"
    on_policy = args.algorithm in {"ppo", "tstm_ppo"}
    if on_policy:
        from algorithms.ppo import RolloutBuffer
        rollout_buffer = RolloutBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.rollout_length,
        )
        replay_buffer = None
    else:
        rollout_buffer = None
        replay_buffer = utils.ReplayBuffer(
            obs_shape=env.observation_space.shape,
            action_shape=env.action_space.shape,
            capacity=args.train_steps,
            batch_size=args.batch_size,
        )
    cropped_obs_shape = (
        3 * args.frame_stack,
        args.image_crop_size,
        args.image_crop_size,
    )
    print("Observations:", env.observation_space.shape)
    print("Cropped observations:", cropped_obs_shape)
    agent = make_agent(
        obs_shape=cropped_obs_shape, action_shape=env.action_space.shape, args=args
    )

    online_gt_total_steps = 50 * int(args.episode_length)
    online_gt_start_step = int(getattr(args, "init_steps", 0))
    online_gt_end_step = int(online_gt_start_step) + int(online_gt_total_steps)

    start_step, episode, episode_reward, done = 0, 0, 0, True
    L = Logger(work_dir)
    start_time = time.time()
    global_start_time = start_time
    total_eval_time = 0.0
    total_eval_steps = 0
    try:
        for step in range(start_step, args.train_steps + 1):
            if done:
                if step > start_step:
                    duration = time.time() - start_time
                    L.log("train/duration", duration, step)
                    if duration > 0:
                        L.log("train/steps_per_sec", episode_step / duration, step)
                    L.log("train/wall_time", time.time() - global_start_time, step)
                    L.log(
                        "train/wall_time_no_eval",
                        time.time() - global_start_time - total_eval_time,
                        step,
                    )
                    L.log("train/total_eval_time", total_eval_time, step)
                    start_time = time.time()
                    L.dump(step)

                # Evaluate agent periodically
                if step > start_step and step % args.eval_freq == 0:
                    print("Evaluating:", work_dir)
                    L.log("eval/episode", episode, step)
                    eval_start_time = time.time()
                    _, eval_steps = evaluate(
                        env, agent, args.algorithm, video, args.eval_episodes, L, step
                    )
                    eval_duration = time.time() - eval_start_time
                    total_eval_time += eval_duration
                    total_eval_steps += int(eval_steps)
                    L.log("eval/duration", eval_duration, step)
                    L.log("eval/steps", int(eval_steps), step)
                    if eval_duration > 0:
                        L.log("eval/steps_per_sec", float(eval_steps) / eval_duration, step)
                    if test_envs is not None:
                        for test_env, test_env_mode in zip(test_envs, test_envs_mode):
                            eval_start_time = time.time()
                            _, test_eval_steps = evaluate(
                                test_env,
                                agent,
                                args.algorithm,
                                video,
                                args.eval_episodes,
                                L,
                                step,
                                test_env=True,
                                eval_mode=test_env_mode,
                            )
                            test_eval_duration = time.time() - eval_start_time
                            total_eval_time += test_eval_duration
                            total_eval_steps += int(test_eval_steps)
                            _suffix = f"_test_env_{test_env_mode}"
                            L.log(f"eval/duration{_suffix}", test_eval_duration, step)
                            L.log(f"eval/steps{_suffix}", int(test_eval_steps), step)
                            if test_eval_duration > 0:
                                L.log(
                                    f"eval/steps_per_sec{_suffix}",
                                    float(test_eval_steps) / test_eval_duration,
                                    step,
                                )
                    L.log("eval/total_duration", total_eval_time, step)
                    L.log("eval/total_steps", int(total_eval_steps), step)
                    L.log("eval/wall_time", time.time() - global_start_time, step)
                    L.dump(step)
                    start_time = time.time()

                # Save agent periodically
                if step > start_step and step % args.save_freq == 0:
                    torch.save(
                        agent.actor.state_dict(), os.path.join(model_dir, f"actor_{step}.pt")
                    )
                    torch.save(
                        agent.critic.state_dict(), os.path.join(model_dir, f"critic_{step}.pt")
                    )
                    if args.algorithm == "sgsac":
                        torch.save(
                            agent.attribution_predictor.state_dict(), os.path.join(model_dir, f"attrib_predictor_{step}.pt")
                        )
                    if args.algorithm in {"madi", "madi_compare", "madi_compare_online"}:
                        torch.save(
                            agent.masker.state_dict(), os.path.join(model_dir, f"masker_{step}.pt")
                        )


                L.log("train/episode_reward", episode_reward, step)
                
                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1

                L.log("train/episode", episode, step)

            # Sample action for data collection
            if step < args.init_steps:
                action = env.action_space.sample()
                logp, value = 0.0, 0.0
            else:
                with utils.eval_mode(agent):
                    if on_policy:
                        action, logp, value = agent.act(obs)
                    else:
                        action = agent.sample_action(obs)
                        logp, value = 0.0, 0.0

            # Take step
            next_obs, reward, done, _ = env.step(action)

            if args.algorithm == "madi_compare_online" and hasattr(agent, "set_online_gt"):
                if step >= online_gt_start_step and step < online_gt_end_step:
                    try:
                        physics = _get_physics_from_env(env)
                        if physics is None:
                            agent.set_online_gt(None, None)
                        else:
                            seg_image = physics.render(
                                height=args.image_size,
                                width=args.image_size,
                                camera_id=0,
                                segmentation=True,
                            )
                            gt_mask = _extract_agent_mask(seg_image)
                            next_obs_np = np.array(next_obs)
                            gt_frame = next_obs_np[-3:, :, :]
                            gt_frame_t = torch.as_tensor(gt_frame).unsqueeze(0).cuda().float()
                            gt_mask_t = torch.as_tensor(gt_mask).unsqueeze(0).unsqueeze(0).cuda().float()
                            agent.set_online_gt(gt_frame_t, gt_mask_t)
                    except Exception:
                        agent.set_online_gt(None, None)
                else:
                    agent.set_online_gt(None, None)

            done_bool = 0 if episode_step + 1 == env._max_episode_steps else float(done)

            if on_policy:
                episode_done = float(done)
                terminal = done_bool
                with torch.no_grad():
                    with utils.eval_mode(agent):
                        next_value = agent.get_value(next_obs)
                rollout_buffer.add(obs, action, logp, reward, episode_done, terminal, value, next_value)
                if rollout_buffer.full:
                    agent.update(rollout_buffer, L, step)
            else:
                replay_buffer.add(obs, action, reward, next_obs, done_bool)

                # Run training update
                if step >= args.init_steps:
                    num_updates = args.init_steps if step == args.init_steps else 1
                    for _ in range(num_updates):
                        agent.update(replay_buffer, L, step)

            episode_reward += reward
            obs = next_obs
            
            episode_step += 1
        total_elapsed = time.time() - global_start_time
        total_train_elapsed = total_elapsed - total_eval_time
        L.log("train/total_wall_time", total_elapsed, args.train_steps)
        L.log("train/total_eval_time", total_eval_time, args.train_steps)
        L.log("train/total_train_time", total_train_elapsed, args.train_steps)
        L.log("train/total_steps", int(args.train_steps), args.train_steps)
        if total_train_elapsed > 0:
            L.log(
                "train/overall_steps_per_sec",
                float(args.train_steps) / total_train_elapsed,
                args.train_steps,
            )
        if total_eval_time > 0:
            L.log(
                "eval/overall_steps_per_sec",
                float(total_eval_steps) / total_eval_time,
                args.train_steps,
            )
        L.log("eval/total_duration", total_eval_time, args.train_steps)
        L.log("eval/total_steps", int(total_eval_steps), args.train_steps)
        L.dump(args.train_steps)
        print("Completed training for", work_dir)
    finally:
        try:
            augmentations.shutdown_places_loader()
        except Exception:
            pass


if __name__ == "__main__":
    args = parse_args()
    main(args)
