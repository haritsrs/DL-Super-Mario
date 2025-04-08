# 🕹️ Super Mario RL Agent

Train a reinforcement learning agent to beat Super Mario Bros using PPO + reward shaping + vectorized environments.

---

## 🧠 Features

- Complex movement (run, jump, duck, etc)
- Reward shaping for progress, success, and penalty on death
- TensorBoard logging
- Parallel environment training (8 envs)
- Video evaluation & checkpoints

---

## ⚙️ Setup

### 1. Create Virtual Environment

# Windows
python -m venv mario-rl
mario-rl\Scripts\activate

# macOS/Linux
python3 -m venv mario-rl
source mario-rl/bin/activate

### 2. Install Dependencies
pip install -r requirements.txt


## 🚀 Training
python -m scripts.train

## 🎮 Evaluate Agent
python -m scripts.evaluate

## 📈 TensorBoard
tensorboard --logdir=./tensorboard_logs
