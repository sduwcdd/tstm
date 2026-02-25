#!/bin/bash
# Single-K RL script for walker_walk (tstm):
# - Uses an existing student checkpoint produced by the segmentation pipeline.
# - Runs RL for 3 seeds (default: 40 41 42)
#
# Usage example:
#   CUDA_VISIBLE_DEVICES=0 K=4 bash src/tstm_segment/run_single_k_rl_walker_walk.sh

set -e

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# ==================== CONFIG ====================
DOMAIN="${DOMAIN:-walker}"
TASK="${TASK:-walk}"

# K: frame stack (must be provided)
K="${K:-}"
if [ -z "$K" ]; then
  echo "Error: please provide K, e.g. K=4"
  exit 1
fi

# Segmentation training seed (used to locate the student ckpt)
SEG_SEED="${SEG_SEED:-0}"

# RL
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-500k}"

# 3 parallel jobs by default (configurable)
SEED0="${SEED0:-40}"
SEED1="${SEED1:-41}"
SEED2="${SEED2:-42}"

GPU0="${GPU0:-0}"
GPU1="${GPU1:-1}"
GPU2="${GPU2:-2}"

# Root log dir for this K (must match the segmentation pipeline output root)
LOG_ROOT="${LOG_ROOT:-logs_k_ablation}"
RUN_NAME="${RUN_NAME:-k${K}}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"

# Student checkpoint path
STUDENT_HIDDEN="${STUDENT_HIDDEN:-32}"
STUDENT_SAVE_DIR="${LOG_DIR}/student_h${STUDENT_HIDDEN}/${DOMAIN}_${TASK}/seed_${SEG_SEED}"
STUDENT_CKPT="${STUDENT_SAVE_DIR}/best_model.pth"

echo "=========================================="
echo "TSTM K Ablation - RL (Single K)"
echo "=========================================="
echo "Env: ${DOMAIN}_${TASK}"
echo "K: ${K}"
echo "SEG_SEED: ${SEG_SEED}"
echo "RL jobs:"
echo "  seed=${SEED0} on GPU=${GPU0}"
echo "  seed=${SEED1} on GPU=${GPU1}"
echo "  seed=${SEED2} on GPU=${GPU2}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "LOG_DIR: ${LOG_DIR}"
echo "Student ckpt: ${STUDENT_CKPT}"
echo "=========================================="
echo ""

if [ ! -f "${STUDENT_CKPT}" ]; then
  echo "Error: student checkpoint not found: ${STUDENT_CKPT}"
  echo "Hint: run segmentation pipeline first:"
  echo "  K=${K} bash src/tstm_segment/run_single_k_seg_pipeline_walker_walk.sh"
  exit 1
fi

mkdir -p "${LOG_DIR}/rl"

echo ""
echo "[RL] Launching 3 parallel nohup jobs ..."

CUDA_VISIBLE_DEVICES="${GPU0}" nohup python -u src/train.py --domain_name "${DOMAIN}" --task_name "${TASK}" --seed "${SEED0}" --algorithm tstm --frame_stack "${K}" --temporal_model_path "${STUDENT_CKPT}" --train_steps "${RL_TRAIN_STEPS}" --log_dir "${LOG_DIR}/rl" > "${LOG_DIR}/rl/nohup_seed_${SEED0}_gpu_${GPU0}.log" 2>&1 &
CUDA_VISIBLE_DEVICES="${GPU1}" nohup python -u src/train.py --domain_name "${DOMAIN}" --task_name "${TASK}" --seed "${SEED1}" --algorithm tstm --frame_stack "${K}" --temporal_model_path "${STUDENT_CKPT}" --train_steps "${RL_TRAIN_STEPS}" --log_dir "${LOG_DIR}/rl" > "${LOG_DIR}/rl/nohup_seed_${SEED1}_gpu_${GPU1}.log" 2>&1 &
CUDA_VISIBLE_DEVICES="${GPU2}" nohup python -u src/train.py --domain_name "${DOMAIN}" --task_name "${TASK}" --seed "${SEED2}" --algorithm tstm --frame_stack "${K}" --temporal_model_path "${STUDENT_CKPT}" --train_steps "${RL_TRAIN_STEPS}" --log_dir "${LOG_DIR}/rl" > "${LOG_DIR}/rl/nohup_seed_${SEED2}_gpu_${GPU2}.log" 2>&1 &

echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "K: ${K}"
echo "Student: ${STUDENT_CKPT}"
echo "RL logs: ${LOG_DIR}/rl/${DOMAIN}_${TASK}/tstm/<seed>/"
echo "nohup logs: ${LOG_DIR}/rl/nohup_seed_<seed>_gpu_<gpu>.log"
