# TSTM: Temporal Segmentation + RL (robot_env)

This document explains how to collect temporal segmentation data in robot_env, train the temporal model (ConvLSTM), optionally distill a student, and finally train RL with the TSTM algorithm.

- Stage 1: Collect ground-truth (GT) masks via depth-based segmentation (robot_env only)
- Stage 2: Train a Teacher temporal segmentation model (ConvLSTM)
- Stage 3: Distill a small Student model
- RL: Use the trained temporal model in TSTM

Key scripts:
- Stage 1 (GT masks): `src/tstm_segment/stage1_collect_gt_depth_robot.py`
- Temporal model: `src/models/video_segmentation.py`
- Stage 2 trainer: `src/tstm_segment/stage2_train_vos_model.py`
- Distillation: `src/tstm_segment/distill_vos_model.py`
- TSTM (RL): `src/algorithms/tstm.py`, `src/train.py`, `src/utils.py`

All paths in this doc are relative to the project root `robot_env/`.

---
## 1) Environment Setup

Recommended (examples; adjust versions to your CUDA/OS). You may also use the provided `setup/conda.yaml` as reference.
Identical to DMC/README.md
```bash
conda create -n vrl python=3.9.21
conda activate vrl
```
**Different from DMC, we use mujoco210 in robot_env,which need some extra install steps**
```bash
pip uninstall -y mujoco_py
pip install "Cython>=0.29.36,<3"
conda install -c conda-forge -y patchelf

pip install -U 'mujoco-py<2.2,>=2.1'
wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
mv mujoco210-linux-x86_64.tar.gz ~/.mujoco/
cd ~/.mujoco/
tar -xvf mujoco210-linux-x86_64.tar.gz
rm mujoco210-linux-x86_64.tar.gz
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/your_path_to_mujoco/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export MUJOCO_GL=egl
export MUJOCO_PY_MUJOCO_PATH=$HOME/.mujoco/mujoco210
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
python -c "import mujoco_py"
```



Notes:

- MuJoCo key path and EGL are set in `src/train.py` (Linux).
- For Places365 overlay augmentation, configure dataset paths in `setup/config.cfg` (key `datasets`).

---

## 2) Quick Start (bash)

Run the full VOS pipeline (collect GT masks → train Teacher → distill Student) with one script:

```bash
PYTHONPATH=src bash src/tstm_segment/train_vos_pipeline_robot.sh
```

Override options via environment variables (examples):

```bash
# Task variants: reach | push | pegbox | hammerall
TASK=push SEED=1  ACTION_SPACE=xy TEACHER_HIDDEN=256 STUDENT_HIDDEN=32 SEQUENCE_LENGTH=5 bash src/tstm_segment/train_vos_pipeline_robot.sh
```

Outputs:
- Teacher: `logs/teacher_h${TEACHER_HIDDEN}/robot_${TASK}/seed_${SEED}/best_model.pth`
- Student: `logs/student_h${STUDENT_HIDDEN}/robot_${TASK}/seed_${SEED}/best_model.pth`

Use the trained model in RL (TSTM):
```bash
python src/train.py --domain_name robot --task_name reach --seed 1 --frame_stack 5 --algorithm tstm --temporal_model_path logs/student_h${STUDENT_HIDDEN}/robot_${TASK}/seed_${SEED}/best_model.pth --train_steps 250k
```
---


## 3) Or Run Stages Manually (Same as 2)

### 3.1 Stage 1: Collect GT Masks (Depth-based)

Use depth-based segmentation to generate binary masks for each frame.

Script: `src/tstm_segment/stage1_collect_gt_depth_robot.py`

```bash
python src/tstm_segment/stage1_collect_gt_depth_robot.py \
  --domain_name robot \
  --task_name reach \
  --seed 0 \
  --frame_stack 5 \
  --episode_length 50 \
  --n_substeps 20 \
  --image_size 84 \
  --camera third_person \
  --action_space xy \
  --collect_episodes 50
```

Output directory:
- `logs/vos_training_data_gt/robot_<task>/seed_<seed>`
  - `frames/episode_XXXX_step_YYYY.png`
  - `masks/episode_XXXX_step_YYYY.npy` and `_vis.png`
  - `episodes_metadata.pkl`

Tips:
- Tasks: `reach`, `push`, `pegbox`, `hammerall`
- Action spaces: `xy` (reach/push), `xyz` (pegbox/hammerall)

---

### 3.2 Stage 2: Train Teacher (ConvLSTM)

Script: `src/tstm_segment/stage2_train_vos_model.py`

- Sequence length must be 5.
- Supported `--conv_lstm_hidden`: `{32, 256}` (use `256` for teacher).
- Optional overlay augmentation samples backgrounds from Places365 (configure paths in `setup/config.cfg`).

```bash
python src/tstm_segment/stage2_train_vos_model.py \
  --data_dir logs/vos_training_data_gt/robot_reach/seed_0 \
  --model_type conv_lstm \
  --sequence_length 5 \
  --batch_size 128 \
  --num_epochs 50 \
  --learning_rate 1e-4 \
  --conv_lstm_hidden 256 \
  --use_overlay \
  --overlay_prob 0.5 \
  --overlay_alpha 0.5
```

- Default save dir: `<data_dir>/../temporal_model` (printed by the script). Best checkpoint: `best_model.pth`.

---

### 3.3 Stage 3: Distill to Student (Optional)

Script: `src/tstm_segment/distill_vos_model.py`

- Teacher: pass `--teacher_path` to `best_model.pth` from Stage 2.
- Student hidden dims: `{32, 256}` (use `32` for lightweight student).
- Distillation uses BCE+Dice vs GT and soft targets from teacher (temperature scaling).

```bash
python src/tstm_segment/distill_vos_model.py \
  --data_dir logs/vos_training_data_gt/robot_reach/seed_0 \
  --teacher_path logs/vos_training_data_gt/robot_reach/temporal_model/best_model.pth \
  --student_hidden 32 \
  --batch_size 128 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --distill_temp 4.0 \
  --distill_alpha 0.7 \
  --use_overlay \
  --overlay_prob 0.5 \
  --overlay_alpha 0.5 \
  --save_dir logs/temporal_distilled/robot_reach/seed_0
```

- Best student is saved to `<save_dir>/best_model.pth`.

---

## 4) RL Training with TSTM

TSTM consumes a 5-frame stack. A trained temporal model checkpoint is required (`--temporal_model_path`).

Key args from `src/arguments.py`:
- `--algorithm tstm`
- `--frame_stack 5` (must be 5)
- `--image_size 84`
- `--episode_length 50`
- `--n_substeps 20` (robot_env uses physics substeps; no action repeat)
- `--train_steps 250k` (replay buffer size equals train steps)
- `--policy_consistency_weight 2` (default in arguments)

Command example (reach):
```bash
python src/train.py \
  --domain_name robot \
  --task_name reach \
  --seed 0 \
  --frame_stack 5 \
  --image_size 84 \
  --episode_length 50 \
  --n_substeps 20 \
  --action_space xy \
  --cameras 0 \
  --algorithm tstm \
  --temporal_model_path logs/temporal_distilled/robot_reach/seed_0/best_model.pth \
  --train_steps 250k \
  --batch_size 128
```
- Replay buffer capacity is set to `train_steps`. See `src/train.py`.
- `select_action/sample_action` will apply the temporal mask; `frame_stack != 5` will raise an error.

---

## 5) Outputs

- Stage 1 data: `logs/temporal_training_data_gt/{DOMAIN}_{TASK}/seed_{SEED}`
- Teacher model: `logs/teacher_h{TEACHER_HIDDEN}/{DOMAIN}_{TASK}/seed_{SEED}/best_model.pth`
- Student model: `logs/student_h{STUDENT_HIDDEN}/{DOMAIN}_{TASK}/seed_{SEED}/best_model.pth`
- RL logs: under `logs/` according to your training framework conventions

---

## 6) Troubleshooting

- `temporal model path invalid or not provided`: pass a valid `--temporal_model_path` (Teacher/Student `best_model.pth`).
- `expected 5 frames (C=15)`: ensure `--frame_stack 5` in RL.
- Overlay/Places365 not found: set `DMCGB_DATASETS` or configure `setup/config.cfg`.
---