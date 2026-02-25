# TSTM: Temporal Segmentation for Task-related Mask in Visual Reinforcement Learning Generalization

This document explains how to train the temporal segmentation model (ConvLSTM) and use it within RL using the TSTM algorithm.

- Stage 1: Collect ground-truth masks from MuJoCo (via dm_control/dmc2gym)
- Stage 2: Train a Teacher temporal segmentation model
- Stage 3: Distill a small Student model
- RL: Use the trained temporal model in TSTM

Key scripts:
- `src/tstm_segment/stage1_collect_gt_mask_simple.py`
- `src/tstm_segment/stage2_train_vos_model.py`
- `src/tstm_segment/distill_vos_model.py`
- `src/tstm_segment/train_full_pipeline.sh`
- `src/algorithms/tstm.py`
- `src/train.py`

All paths in this doc are relative to the project root `DMC/`.

---



## 1) Environment Setup

If you don't have the right mujoco version installed: 

```bash
sh setup/install_mujoco_deps.sh
sh setup/prepare_dm_control_xp.sh
```

Recommended (examples; adjust versions to your CUDA/OS):

```bash
conda create -n test python=3.9.21
conda activate test

pip install "pip<24.1" 
pip install -U "pip<24.1" "setuptools<66" "wheel<0.39"
# Install PyTorch (example for CUDA 11.8; adjust accordingly)
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu118
# CUDA 11.3
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
# CUDA 10.2
# Common libs
pip install absl-py pyparsing numpy==1.21.0 termcolor imageio imageio-ffmpeg opencv-python==4.10.0.84 xmltodict tqdm einops kornia==0.8.1 captum tensorboard 
pip install gym==0.21.0

# DMControl stack 
# pip install mujoco-py==2.1.2.14
pip install "numpy==1.21.0" "scipy<1.11"
pip install "mujoco==2.1.5" "dm-control==1.0.2"
# mujoco_py==2.0.2.8
# Install dmc2gym wrapper
cd src/env/dmc2gym
pip install -e . --no-deps
cd ../../..

# verify installation
python -c "import gym; import dm_control; import mujoco; import dmc2gym; print('ok', gym.__version__)"
```
pip install PyOpenGL PyOpenGL.accelerate

# Datasets
Overlay augmentation for Stage 2/3 uses Places365. Download and configure one of the following:
wget http://data.csail.mit.edu/places/places365/places365standard_easyformat.tar
mkdir -p dataset
tar -xvf dataset/places365standard_easyformat.tar -C dataset
After downloading and extracting the data, add your dataset directory to the datasets list in `setup/config.cfg`.
set an environment variable:
```bash
export DMCGB_DATASETS=/data/duweicheng/tstm/TSTM_on_DMC/dataset
```

note: The configuration of Gym==0.21.0 and the subsequent achievement of successful rendering can prove to be a challenging task for novice users. In the event of encountering errors, it is advisable to seek relevant assistance via the issues section on Gym's GitHub repository. With regard to any system packages requiring installation, it is necessary to request permission from the server administrator.


## 2) Quick Start (bash)

Runs all stages with minimal logs.

```bash
PYTHONPATH=src bash src/tstm_segment/train_full_pipeline.sh
```

Optional environment variables (defaults from the script):
- `DOMAIN=walker`
- `TASK=walk`
- `SEED=0`
- `SEQUENCE_LENGTH=5`
- `TEACHER_HIDDEN=256`
- `STUDENT_HIDDEN=32`
- `TEACHER_EPOCHS=100`
- `STUDENT_EPOCHS=100`
- `OVERLAY_ALPHA=0.5`
- `OVERLAY_PROB=0.5`
- `NUM_WORKERS=16`

Outputs:
- Stage 1 data: `logs/temporal_training_data_gt/{DOMAIN}_{TASK}/seed_{SEED}`
- Teacher model: `logs/teacher_h{TEACHER_HIDDEN}/{DOMAIN}_{TASK}/seed_{SEED}/best_model.pth`
- Student model: `logs/student_h{STUDENT_HIDDEN}/{DOMAIN}_{TASK}/seed_{SEED}/best_model.pth`

---

## 3) Or Run Stages Manually (Same as 2)

### 3.1 Stage 1: Collect GT masks

```bash
python src/tstm_segment/stage1_collect_gt_mask_simple.py \
  --domain_name walker \
  --task_name walk \
  --seed 0 \
  --collect_episodes 50 \
  --log_dir logs \
  --image_size 84 \
  --frame_stack 8 \
  --action_repeat 4 \
  --episode_length 1000
```

Output data directory:
- `logs/temporal_training_data_gt/walker_walk/seed_0`

### 3.2 Stage 2: Train Teacher (ConvLSTM)

- Sequence length must be 5.
- Supported `hidden_dim`: {32, 256}.

```bash
python src/tstm_segment/stage2_train_vos_model.py \
  --data_dir logs/temporal_training_data_gt/walker_walk/seed_0 \
  --model_type conv_lstm \
  --sequence_length 5 \
  --batch_size 128 \
  --num_epochs 100 \
  --learning_rate 1e-4 \
  --conv_lstm_hidden 256 \
  --num_workers 16 \
  --save_dir logs/teacher_h256/walker_walk/seed_0 \
  --use_overlay \
  --overlay_alpha 0.5 \
  --overlay_prob 0.5
```

### 3.3 Stage 3: Distill Student

- Supported `hidden_dim`: {32, 256}.

```bash
python src/tstm_segment/distill_vos_model.py \
  --data_dir logs/temporal_training_data_gt/walker_walk/seed_0 \
  --teacher_path logs/teacher_h256/walker_walk/seed_0/best_model.pth \
  --teacher_hidden 256 \
  --student_hidden 32 \
  --sequence_length 5 \
  --batch_size 128 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --distill_temp 4.0 \
  --distill_alpha 0.7 \
  --use_overlay \
  --overlay_alpha 0.5 \
  --overlay_prob 0.5 \
  --num_workers 16 \
  --save_dir logs/student_h32/walker_walk/seed_0
```

---

## 4) RL Training with TSTM

TSTM consumes a 5-frame stack. A trained temporal model checkpoint is required (`--temporal_model_path`).

Key args from `src/arguments.py`:
- `--algorithm tstm`
- `--temporal_model_path /path/to/best_model.pth`
- `--frame_stack 5` (strictly required by TSTM; default is 5)
- `--eval_mode` in `{train, color_easy, color_hard, video_easy, video_hard, distracting_cs, all, both, none}`

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python src/train.py --algorithm tstm --domain_name walker --task_name walk --seed 0 --frame_stack 5 --temporal_model_path logs/student_h32/walker_walk/seed_0/best_model.pth --eval_mode both --train_steps 10000 --eval_freq 1000
```

Notes:
- You may use the Teacher checkpoint instead of Student by pointing `--temporal_model_path` to the Teacher model.
- If `frame_stack != 5`, TSTM raises an error, e.g. `expected 5 frames (C=15)`.
- The temporal model runs in eval mode and is frozen during RL.

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