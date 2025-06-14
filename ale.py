import gymnasium as gym
import ale_py
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
import os

# Register ALE environments
gym.register_envs(ale_py)

# ======= ENV CREATION FUNCTIONS ======= #

def make_train_env():
    def _init():
        env = gym.make("ALE/SpaceInvaders-v5")
        env = ResizeObservation(env, (84, 84))
        env = GrayscaleObservation(env, keep_dim=True)
        return env
    return _init

def make_eval_env():
    env = gym.make("ALE/SpaceInvaders-v5")
    env = ResizeObservation(env, (84, 84))
    env = GrayscaleObservation(env, keep_dim=True)
    return env

# ========== MAIN ========== #

if __name__ == "__main__":
    num_envs = 8
    env_fns = [make_train_env() for _ in range(num_envs)]
    train_env = SubprocVecEnv(env_fns)
    train_env = VecFrameStack(train_env, n_stack=4)

    # Create eval env (now using DummyVecEnv)
    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # Log directories
    log_dir = "logs/ppo_space_invaders"
    os.makedirs(log_dir, exist_ok=True)
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # PPO Model
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cuda"
    )

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(log_dir, "best_model"),
        log_path=os.path.join(log_dir, "eval"),
        eval_freq=100_000,
        deterministic=True,
        render=False
    )

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000 // num_envs,
        save_path=checkpoint_dir,
        name_prefix="ppo_checkpoint"
    )

    # Train with both callbacks
    model.learn(
        total_timesteps=5_000_000,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )

    # Save final model
    model.save("ppo_spaceinvaders_model")

    # Close environments
    train_env.close()
    eval_env.close()
