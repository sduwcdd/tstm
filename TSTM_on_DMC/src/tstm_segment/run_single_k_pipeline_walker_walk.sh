#!/bin/bash
# Single-K pipeline for walker_walk:
# 1) collect GT (if missing)
# 2) train teacher (if missing)
# 3) distill student (if missing)
# 4) train RL (tstm) for 3 seeds using that student
#
# Usage example (run concurrently in different terminals):
#   CUDA_VISIBLE_DEVICES=0 K=1 bash src/tstm_segment/run_single_k_pipeline_walker_walk.sh
#   CUDA_VISIBLE_DEVICES=1 K=2 bash src/tstm_segment/run_single_k_pipeline_walker_walk.sh
#   CUDA_VISIBLE_DEVICES=2 K=4 bash src/tstm_segment/run_single_k_pipeline_walker_walk.sh
#   CUDA_VISIBLE_DEVICES=3 K=8 bash src/tstm_segment/run_single_k_pipeline_walker_walk.sh

set -e

export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

# ==================== CONFIG ====================
DOMAIN="${DOMAIN:-walker}"
TASK="${TASK:-walk}"

# K: sequence length / frame stack (must be provided)
K="${K:-}"
if [ -z "$K" ]; then
  echo "Error: please provide K, e.g. K=1"
  exit 1
fi

# Segmentation training seed (shared across RL seeds)
SEG_SEED="${SEG_SEED:-0}"

# Data collection
COLLECT_EPISODES="${COLLECT_EPISODES:-50}"

# Teacher
TEACHER_HIDDEN="${TEACHER_HIDDEN:-256}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-100}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-128}"
TEACHER_LR="${TEACHER_LR:-1e-3}"

# Student distillation
STUDENT_HIDDEN="${STUDENT_HIDDEN:-32}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-100}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-128}"
STUDENT_LR="${STUDENT_LR:-1e-3}"
DISTILL_TEMP="${DISTILL_TEMP:-4.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.7}"

# Data augmentation (used in stage 2/3)
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.5}"
OVERLAY_PROB="${OVERLAY_PROB:-0.5}"

# Misc
NUM_WORKERS="${NUM_WORKERS:-16}"

# RL
RL_SEEDS="${RL_SEEDS:-40 41 42}"
RL_LOG_ROOT="${RL_LOG_ROOT:-logs_k_ablation}"
RL_TRAIN_STEPS="${RL_TRAIN_STEPS:-500k}"

# Root log dir for this K (to avoid override across K)
LOG_ROOT="${LOG_ROOT:-logs_k_ablation}"
RUN_NAME="${RUN_NAME:-k${K}}"
LOG_DIR="${LOG_ROOT}/${RUN_NAME}"

# Paths
DATA_DIR="${LOG_DIR}/temporal_training_data_gt/${DOMAIN}_${TASK}/seed_${SEG_SEED}"
TEACHER_SAVE_DIR="${LOG_DIR}/teacher_h${TEACHER_HIDDEN}/${DOMAIN}_${TASK}/seed_${SEG_SEED}"
STUDENT_SAVE_DIR="${LOG_DIR}/student_h${STUDENT_HIDDEN}/${DOMAIN}_${TASK}/seed_${SEG_SEED}"
STUDENT_CKPT="${STUDENT_SAVE_DIR}/best_model.pth"

echo "=========================================="
echo "TSTM K Ablation (Single K) - walker_walk"
echo "=========================================="
echo "Env: ${DOMAIN}_${TASK}"
echo "K: ${K}"
echo "SEG_SEED: ${SEG_SEED}"
echo "RL_SEEDS: ${RL_SEEDS}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "LOG_DIR: ${LOG_DIR}"
echo "DATA_DIR: ${DATA_DIR}"
echo "TEACHER_SAVE_DIR: ${TEACHER_SAVE_DIR}"
echo "STUDENT_SAVE_DIR: ${STUDENT_SAVE_DIR}"
echo "=========================================="
echo ""

# ==================== Step 1: Collect GT ====================
echo "Step 1/4: Collect GT"

if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/episodes_metadata.pkl" ]; then
  echo "Data exists: $DATA_DIR"
  echo "Skip collection"
  COLLECT_DATA=false
else
  echo "Data not found, will collect"
  COLLECT_DATA=true
fi

if [ "$COLLECT_DATA" = true ]; then
  echo "Collecting GT..."
  python src/tstm_segment/stage1_collect_gt_mask_simple.py \
    --domain_name $DOMAIN \
    --task_name $TASK \
    --seed $SEG_SEED \
    --collect_episodes $COLLECT_EPISODES \
    --log_dir ${LOG_DIR} \
    --image_size 84 \
    --frame_stack $K \
    --action_repeat 4 \
    --episode_length 1000
  echo ""
  echo "GT collection done"
  echo "Saved to: ${DATA_DIR}"
fi

echo ""
# ==================== Step 2: Train Teacher ====================
echo "Step 2/4: Train Teacher"

if [ -f "${TEACHER_SAVE_DIR}/best_model.pth" ]; then
  echo "Teacher checkpoint exists, skip"
else
  python src/tstm_segment/stage2_train_vos_model.py \
    --data_dir "${DATA_DIR}" \
    --model_type conv_lstm \
    --sequence_length $K \
    --batch_size $TEACHER_BATCH_SIZE \
    --num_epochs $TEACHER_EPOCHS \
    --learning_rate $TEACHER_LR \
    --conv_lstm_hidden $TEACHER_HIDDEN \
    --num_workers $NUM_WORKERS \
    --save_dir "${TEACHER_SAVE_DIR}" \
    --use_overlay \
    --overlay_alpha $OVERLAY_ALPHA \
    --overlay_prob $OVERLAY_PROB
fi

echo ""
# ==================== Step 3: Distill Student ====================
echo "Step 3/4: Distill Student"
echo "Teacher: ${TEACHER_SAVE_DIR}/best_model.pth"

if [ -f "${STUDENT_CKPT}" ]; then
  echo "Student checkpoint exists, skip"
else
  python src/tstm_segment/distill_vos_model.py \
    --data_dir "${DATA_DIR}" \
    --teacher_path "${TEACHER_SAVE_DIR}/best_model.pth" \
    --teacher_hidden $TEACHER_HIDDEN \
    --student_hidden $STUDENT_HIDDEN \
    --sequence_length $K \
    --batch_size $STUDENT_BATCH_SIZE \
    --num_epochs $STUDENT_EPOCHS \
    --learning_rate $STUDENT_LR \
    --distill_temp $DISTILL_TEMP \
    --distill_alpha $DISTILL_ALPHA \
    --use_overlay \
    --overlay_alpha $OVERLAY_ALPHA \
    --overlay_prob $OVERLAY_PROB \
    --num_workers $NUM_WORKERS \
    --save_dir "${STUDENT_SAVE_DIR}"
fi

echo ""
# ==================== Step 4: RL Training (3 seeds) ====================
echo "Step 4/4: RL Training (tstm)"
echo "Student ckpt: ${STUDENT_CKPT}"

if [ ! -f "${STUDENT_CKPT}" ]; then
  echo "Error: student checkpoint not found: ${STUDENT_CKPT}"
  exit 1
fi

for SEED in ${RL_SEEDS}; do
  echo ""
  echo "[RL] Running seed=${SEED} ..."
  python src/train.py \
    --domain_name ${DOMAIN} \
    --task_name ${TASK} \
    --seed ${SEED} \
    --algorithm tstm \
    --frame_stack ${K} \
    --temporal_model_path "${STUDENT_CKPT}" \
    --train_steps "${RL_TRAIN_STEPS}" \
    --log_dir "${LOG_DIR}/rl"
done

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "K: ${K}"
echo "Teacher: ${TEACHER_SAVE_DIR}/best_model.pth"
echo "Student: ${STUDENT_CKPT}"
echo "RL logs: ${LOG_DIR}/rl/${DOMAIN}_${TASK}/tstm/<seed>/"
