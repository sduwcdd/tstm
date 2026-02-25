#!/bin/bash
set -e


# ==================== arguments ====================
TASK="${TASK:-reach}"  # reach, push, pegbox, hammerall
SEED="${SEED:-0}"

COLLECT_EPISODES="${COLLECT_EPISODES:-50}"

TEACHER_HIDDEN="${TEACHER_HIDDEN:-256}"
TEACHER_EPOCHS="${TEACHER_EPOCHS:-100}"
TEACHER_BATCH_SIZE="${TEACHER_BATCH_SIZE:-128}"
TEACHER_LR="${TEACHER_LR:-1e-3}"

STUDENT_HIDDEN="${STUDENT_HIDDEN:-32}"
STUDENT_EPOCHS="${STUDENT_EPOCHS:-100}"
STUDENT_BATCH_SIZE="${STUDENT_BATCH_SIZE:-128}"
STUDENT_LR="${STUDENT_LR:-1e-3}"
DISTILL_TEMP="${DISTILL_TEMP:-4.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.7}"

OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.5}"
OVERLAY_PROB="${OVERLAY_PROB:-0.5}"

NUM_WORKERS="${NUM_WORKERS:-16}"
SEQUENCE_LENGTH="${SEQUENCE_LENGTH:-5}"

ACTION_SPACE="${ACTION_SPACE:-xy}"  # reach/push:xy, pegbox/hammerall:xyz
CAMERA="${CAMERA:-third_person}"


DATA_DIR="logs/vos_training_data_gt/robot_${TASK}/seed_${SEED}"
TEACHER_SAVE_DIR="logs/teacher_h${TEACHER_HIDDEN}/robot_${TASK}/seed_${SEED}"
STUDENT_SAVE_DIR="logs/student_h${STUDENT_HIDDEN}/robot_${TASK}/seed_${SEED}"

echo "=========================================="
echo "VOS pipeline (robot_env, depth-based)"
echo "Task=robot_${TASK}  Seed=${SEED}"
echo "Data=${DATA_DIR}  Teacher=${TEACHER_SAVE_DIR}  Student=${STUDENT_SAVE_DIR}"
echo "=========================================="

echo ""
echo "[1/3] Collect GT (depth) -> ${DATA_DIR}"

if [ -d "$DATA_DIR" ] && [ -f "$DATA_DIR/episodes_metadata.pkl" ]; then
  echo "Exists, skip collection"
  COLLECT_DATA=false
else
  echo "Collecting..."
  COLLECT_DATA=true
fi

if [ "$COLLECT_DATA" = true ]; then
  python src/tstm_segment/stage1_collect_gt_depth_robot.py \
    --domain_name robot \
    --task_name $TASK \
    --seed $SEED \
    --collect_episodes $COLLECT_EPISODES \
    --log_dir logs \
    --image_size 84 \
    --frame_stack 1 \
    --episode_length 50 \
    --n_substeps 20 \
    --camera $CAMERA \
    --action_space $ACTION_SPACE
  echo "Done: ${DATA_DIR}"
fi

echo "[2/3] Train Teacher (hidden=${TEACHER_HIDDEN}) -> ${TEACHER_SAVE_DIR}"

if [ -f "${TEACHER_SAVE_DIR}/best_model.pth" ]; then
  echo "Exists, skip"
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

echo "[3/3] Distill Student (hidden=${STUDENT_HIDDEN}) -> ${STUDENT_SAVE_DIR}"

if [ -f "${STUDENT_SAVE_DIR}/best_model.pth" ]; then
  echo "Exists, skip"
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

echo ""
echo "Done. Models:"
echo "  Teacher: ${TEACHER_SAVE_DIR}/best_model.pth"
echo "  Student: ${STUDENT_SAVE_DIR}/best_model.pth"
echo "Use in RL (TSTM):"
echo "  python src/train.py \\
    --algorithm tstm \\
    --temporal_model_path ${STUDENT_SAVE_DIR}/best_model.pth \\
    --task_name ${TASK} \\
    --frame_stack 5 \\
    --train_steps 250k"
