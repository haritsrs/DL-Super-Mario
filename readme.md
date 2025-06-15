# 🕹️ Super Mario RL-37

Train a reinforcement learning agent to beat Super Mario Bros using PPO + reward shaping + vectorized environments.
Why 37? Because the first iteration of the ai that finished was number 37. You'll be in our hearts forever

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
for logging
tensorboard --logdir=./tensorboard_logs
