#!/bin/bash
# Full pipeline: collect GT + train Teacher + distill Student (DMC)
# Output: Student model -> logs/student_h${STUDENT_HIDDEN}/${DOMAIN}_${TASK}/seed_${SEED}/best_model.pth

set -e  # exit on error
# ==================== CONFIG ====================
DOMAIN="${DOMAIN:-walker}"
TASK="${TASK:-walk}"
SEED="${SEED:-0}"

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
OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.5}"   # fixed to 0.5 in DMC impl
OVERLAY_PROB="${OVERLAY_PROB:-0.5}"

# Misc
NUM_WORKERS="${NUM_WORKERS:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-5}"

# Paths (relative to DMC)
DATA_DIR="logs/temporal_training_data_gt/${DOMAIN}_${TASK}/seed_${SEED}"
TEACHER_SAVE_DIR="logs/teacher_h${TEACHER_HIDDEN}/${DOMAIN}_${TASK}/seed_${SEED}"
STUDENT_SAVE_DIR="logs/student_h${STUDENT_HIDDEN}/${DOMAIN}_${TASK}/seed_${SEED}"

echo "=========================================="
echo "DMC Temporal Segmentation Pipeline"
echo "=========================================="
echo "Env: ${DOMAIN}_${TASK}, Seed: ${SEED}"
if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi
echo "Data dir: ${DATA_DIR}"
echo "Teacher out: ${TEACHER_SAVE_DIR}"
echo "Student out: ${STUDENT_SAVE_DIR}"
echo "=========================================="

echo ""
# ==================== Step 1: Collect GT ====================
echo "Step 1/3: Collect GT"

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
    --seed $SEED \
    --collect_episodes $COLLECT_EPISODES \
    --log_dir logs \
    --image_size 84 \
    --frame_stack 3 \
    --action_repeat 4 \
    --episode_length 1000
  echo ""
  echo "GT collection done"
  echo "Saved to: ${DATA_DIR}"
fi

echo ""
# ==================== Step 2: Train Teacher ====================
echo "Step 2/3: Train Teacher"

if [ -f "${TEACHER_SAVE_DIR}/best_model.pth" ]; then
  echo "Teacher checkpoint exists, skip"
else
  python src/tstm_segment/stage2_train_vos_model.py \
    --data_dir "${DATA_DIR}" \
    --model_type conv_lstm \
    --sequence_length $SEQUENCE_LENGTH \
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
echo "Step 3/3: Distill Student"
echo "Teacher: ${TEACHER_SAVE_DIR}/best_model.pth"

if [ -f "${STUDENT_SAVE_DIR}/best_model.pth" ]; then
  echo "Student checkpoint exists, skip"
else
  python src/tstm_segment/distill_vos_model.py \
    --data_dir "${DATA_DIR}" \
    --teacher_path "${TEACHER_SAVE_DIR}/best_model.pth" \
    --teacher_hidden $TEACHER_HIDDEN \
    --student_hidden $STUDENT_HIDDEN \
    --sequence_length $SEQUENCE_LENGTH \
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

# ==================== Summary ====================
echo ""
echo "=========================================="
echo "Done"
echo "=========================================="
echo "Teacher: ${TEACHER_SAVE_DIR}/best_model.pth"
echo "Student: ${STUDENT_SAVE_DIR}/best_model.pth"
