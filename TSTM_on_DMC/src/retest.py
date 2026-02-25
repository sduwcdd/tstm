"""
重测试脚本 - 使用保存的模型在训练期间相同的环境中测试

使用方法:
# 测试训练环境（对应eval.log中的episode_reward）
python retest.py --domain_name walker --task_name walk --algorithm sgsac --seed 1 --model_step 500000 --env_type train

# 测试color_hard环境（对应eval.log中的episode_reward_test_env_color_hard）
python retest.py --domain_name walker --task_name walk --algorithm sgsac --seed 1 --model_step 500000 --env_type color_hard
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import torch
import numpy as np
import gym
import utils
import argparse
from env.wrappers import make_env
from algorithms.factory import make_agent

def evaluate(env, agent, num_episodes):
    """评估函数"""
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0  # 添加步数统计
        
        while not done:
            with torch.no_grad():
                with utils.eval_mode(agent):
                    action = agent.select_action(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1
        
        episode_rewards.append(episode_reward)
        # 显示奖励和步数，帮助诊断问题
        print(f"Episode {i+1}/{num_episodes}: reward={episode_reward}, steps={episode_steps}")
    
    return episode_rewards


def test_single_env(env_mode, env_seed, agent, args, work_dir):
    """测试单个环境并保存结果"""
    print(f"\n{'='*60}")
    print(f"开始测试环境: {env_mode}")
    print(f"环境种子: {env_seed}")
    print(f"{'='*60}")
    
    # 创建环境（不要重新设置全局种子，避免干扰环境内部随机状态）
    # 环境的seed参数会控制环境内部的随机性
    gym.logger.set_level(40)
    env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=env_seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        mode=env_mode,
        intensity=args.distracting_cs_intensity if env_mode == 'distracting_cs' else None
    )
    
    # 开始测试
    print(f"\n开始测试 {args.test_episodes} 个回合...\n")
    rewards = evaluate(env, agent, args.test_episodes)
    
    # 统计结果
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    min_reward = np.min(rewards)
    max_reward = np.max(rewards)
    median_reward = np.median(rewards)
    
    print(f"\n{'='*60}")
    print(f"测试结果统计 - {env_mode}:")
    print(f"  平均奖励: {mean_reward} ± {std_reward}")
    print(f"  最小值: {min_reward}")
    print(f"  最大值: {max_reward}")
    print(f"  中位数: {median_reward}")
    print(f"{'='*60}\n")
    
    # 保存结果
    result_file = os.path.join(work_dir, f"retest_{env_mode}_step{args.model_step}.txt")
    with open(result_file, 'w') as f:
        f.write(f"Environment: {args.domain_name}_{args.task_name}\n")
        f.write(f"Algorithm: {args.algorithm}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Env Seed: {env_seed}\n")
        f.write(f"Env Mode: {env_mode}\n")
        f.write(f"Model Step: {args.model_step}\n")
        f.write(f"Test Episodes: {args.test_episodes}\n\n")
        f.write(f"Mean: {mean_reward}\n")
        f.write(f"Std: {std_reward}\n")
        f.write(f"Min: {min_reward}\n")
        f.write(f"Max: {max_reward}\n")
        f.write(f"Median: {median_reward}\n\n")
        f.write("All rewards:\n")
        for i, r in enumerate(rewards):
            f.write(f"Episode {i+1}: {r}\n")
    
    print(f"结果已保存到: {result_file}")
    
    return {
        'env_mode': env_mode,
        'mean': mean_reward,
        'std': std_reward,
        'min': min_reward,
        'max': max_reward,
        'median': median_reward,
        'all_rewards': rewards
    }


def main():
    parser = argparse.ArgumentParser()
    # 环境参数
    parser.add_argument("--domain_name", required=True, type=str)
    parser.add_argument("--task_name", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--algorithm", required=True, type=str)
    
    # 模型参数
    parser.add_argument("--model_step", required=True, type=int, help="要加载的模型步数")
    parser.add_argument("--log_dir", default="logs", type=str)
    
    # 评估模式：train, color_easy, color_hard, video_easy, video_hard, distracting_cs, both, all
    parser.add_argument("--eval_mode", default="train", type=str, 
                       help="评估模式：train(训练环境), color_hard/video_easy/video_hard(单个测试环境), both(video_easy+video_hard), all(color_hard+video_easy+video_hard)")
    
    # 测试参数
    parser.add_argument("--test_episodes", default=30, type=int, help="测试回合数")
    parser.add_argument("--frame_stack", default=5, type=int)
    parser.add_argument("--action_repeat", default=4, type=int)
    parser.add_argument("--episode_length", default=1000, type=int)
    parser.add_argument("--distracting_cs_intensity", default=0.0, type=float)
    
    # 架构参数（需要与训练时一致）
    parser.add_argument("--num_shared_layers", default=11, type=int)
    parser.add_argument("--num_head_layers", default=0, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    parser.add_argument("--projection_dim", default=100, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    parser.add_argument("--discount", default=0.99, type=float)
    parser.add_argument("--init_temperature", default=0.1, type=float)
    parser.add_argument("--actor_log_std_min", default=-10, type=float)
    parser.add_argument("--actor_log_std_max", default=2, type=float)
    parser.add_argument("--critic_tau", default=0.01, type=float)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--actor_update_freq", default=2, type=int)
    parser.add_argument("--critic_target_update_freq", default=2, type=int)
    parser.add_argument("--sgqn_quantile", default=0.90, type=float)
    
    # Actor参数
    parser.add_argument("--actor_lr", default=1e-3, type=float)
    parser.add_argument("--actor_beta", default=0.9, type=float)
    
    # Critic参数
    parser.add_argument("--critic_lr", default=1e-3, type=float)
    parser.add_argument("--critic_beta", default=0.9, type=float)
    parser.add_argument("--critic_weight_decay", default=0, type=float)
    
    # 熵参数
    parser.add_argument("--alpha_lr", default=1e-4, type=float)
    parser.add_argument("--alpha_beta", default=0.5, type=float)
    
    # 辅助任务参数
    parser.add_argument("--aux_lr", default=3e-4, type=float)
    parser.add_argument("--aux_beta", default=0.9, type=float)
    parser.add_argument("--aux_update_freq", default=2, type=int)
    
    # VICReg + IIEC参数
    parser.add_argument("--policy_consistency_weight", default=2.0, type=float)
    parser.add_argument("--byol_loss_weight", default=0.05, type=float)
    parser.add_argument("--vicreg_lambda", default=25.0, type=float)
    parser.add_argument("--vicreg_mu", default=25.0, type=float)
    parser.add_argument("--vicreg_nu", default=0.1, type=float)
    parser.add_argument("--vicreg_gamma", default=1.0, type=float)
    parser.add_argument("--vicreg_warmup_steps", default=0, type=int)
    parser.add_argument("--expander_hidden_dim", default=2048, type=int)
    parser.add_argument("--expander_output_dim", default=2048, type=int)
    
    # VOS checkpoint路径
    parser.add_argument("--vos_model_path", default="", type=str)
    
    # 其他算法参数
    parser.add_argument("--svea_alpha", default=0.5, type=float)
    parser.add_argument("--svea_beta", default=0.5, type=float)
    parser.add_argument("--svea_contrastive_coeff", default=0.1, type=float)
    parser.add_argument("--svea_norm_coeff", default=0.1, type=float)
    parser.add_argument("--attrib_coeff", default=0.25, type=float)
    parser.add_argument("--consistency", default=1, type=int)
    parser.add_argument("--soda_batch_size", default=256, type=int)
    parser.add_argument("--soda_tau", default=0.005, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    
    args = parser.parse_args()
    
    # 设置图像大小
    if args.algorithm in {"rad", "curl", "pad", "soda"}:
        args.image_size = 100
        args.image_crop_size = 84
    else:
        args.image_size = 84
        args.image_crop_size = 84
    
    # 配置环境
    home = os.environ.get("HOME", os.path.expanduser("~"))
    os.environ["MUJOCO_GL"] = "egl"
    
    # 设置种子（与train.py第79行一致，只设置一次）
    utils.set_seed_everywhere(args.seed)
    print(f"\n全局随机种子已设置为: {args.seed}")
    
    # 确定要测试的环境列表（与train.py保持一致）
    test_configs = []  # (env_mode, env_seed, description)
    
    if args.eval_mode == "train":
        # 对应 eval.log 中的 episode_reward
        test_configs.append(("train", args.seed, "训练环境 (对应eval.log中的episode_reward)"))
    elif args.eval_mode == "both":
        # 对应 eval.log 中的 video_easy 和 video_hard
        test_configs.append(("video_easy", args.seed + 42, "测试环境 video_easy (对应eval.log中的episode_reward_test_env_video_easy)"))
        test_configs.append(("video_hard", args.seed + 42, "测试环境 video_hard (对应eval.log中的episode_reward_test_env_video_hard)"))
    elif args.eval_mode == "all":
        # 对应 eval.log 中的 color_hard, video_easy 和 video_hard
        test_configs.append(("color_hard", args.seed + 42, "测试环境 color_hard (对应eval.log中的episode_reward_test_env_color_hard)"))
        test_configs.append(("video_easy", args.seed + 42, "测试环境 video_easy (对应eval.log中的episode_reward_test_env_video_easy)"))
        test_configs.append(("video_hard", args.seed + 42, "测试环境 video_hard (对应eval.log中的episode_reward_test_env_video_hard)"))
    else:
        # 单个测试环境
        test_configs.append((args.eval_mode, args.seed + 42, f"测试环境 {args.eval_mode} (对应eval.log中的episode_reward_test_env_{args.eval_mode})"))
    
    print(f"\n{'='*60}")
    print(f"=== 重测试配置 ===")
    print(f"域: {args.domain_name}_{args.task_name}")
    print(f"算法: {args.algorithm}")
    print(f"原始种子: {args.seed}")
    print(f"模型步数: {args.model_step}")
    print(f"测试回合(每个环境): {args.test_episodes}")
    print(f"评估模式: {args.eval_mode}")
    print(f"将测试 {len(test_configs)} 个环境")
    print(f"{'='*60}")
    
    # 设置工作目录
    work_dir = os.path.join(
        args.log_dir,
        f"{args.domain_name}_{args.task_name}",
        args.algorithm,
        str(args.seed)
    )
    print(f"\n工作目录: {work_dir}")
    
    model_dir = os.path.join(work_dir, "model")
    assert os.path.exists(model_dir), f"模型目录不存在: {model_dir}"
    
    # 创建一个临时环境来获取action_shape（用于创建agent）
    gym.logger.set_level(40)
    temp_env = make_env(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        episode_length=args.episode_length,
        action_repeat=args.action_repeat,
        image_size=args.image_size,
        frame_stack=args.frame_stack,
        mode="train"
    )
    
    # 创建agent
    assert torch.cuda.is_available(), "需要CUDA"
    cropped_obs_shape = (3 * args.frame_stack, args.image_crop_size, args.image_crop_size)
    agent = make_agent(
        obs_shape=cropped_obs_shape,
        action_shape=temp_env.action_space.shape,
        args=args
    )
    del temp_env  # 删除临时环境
    
    # 加载模型
    actor_path = os.path.join(model_dir, f"actor_{args.model_step}.pt")
    critic_path = os.path.join(model_dir, f"critic_{args.model_step}.pt")
    
    print(f"\n加载模型:")
    print(f"  Actor: {actor_path}")
    print(f"  Critic: {critic_path}")
    
    assert os.path.exists(actor_path), f"Actor模型不存在: {actor_path}"
    assert os.path.exists(critic_path), f"Critic模型不存在: {critic_path}"
    
    agent.actor.load_state_dict(torch.load(actor_path))
    agent.critic.load_state_dict(torch.load(critic_path))
    
    # 如果是sgsac，加载attribution_predictor
    if args.algorithm == 'sgsac':
        attrib_path = os.path.join(model_dir, f"attrib_predictor_{args.model_step}.pt")
        print(f"  Attribution Predictor: {attrib_path}")
        if os.path.exists(attrib_path):
            agent.attribution_predictor.load_state_dict(torch.load(attrib_path))
    
    agent.train(False)  # 评估模式
    print("✓ 模型加载成功\n")
    
    # 测试所有配置的环境
    all_results = []
    for env_mode, env_seed, description in test_configs:
        print(f"\n{description}")
        result = test_single_env(env_mode, env_seed, agent, args, work_dir)
        all_results.append(result)
    
    # 打印汇总
    print(f"\n\n{'='*60}")
    print(f"=== 所有环境测试完成 ===")
    print(f"{'='*60}")
    for result in all_results:
        print(f"\n环境: {result['env_mode']}")
        print(f"  平均奖励: {result['mean']} ± {result['std']}")
        print(f"  范围: [{result['min']}, {result['max']}]")
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
