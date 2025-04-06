import gym
import numpy as np
from gym_super_mario_bros import make as make_mario
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from utils.wrappers import SkipFrame, GrayScaleObservation, ResizeObservation, NormalizeObservation


def create_vec_env(n_envs=1, level='SuperMarioBros-1-1-v3'):
    def make_env():
        env = make_mario(level)
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = SkipFrame(env, skip=4)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, shape=84)
        env = NormalizeObservation(env)
        return env

    env_fns = [make_env for _ in range(n_envs)]
    env = DummyVecEnv(env_fns)
    env = VecFrameStack(env, n_stack=4)
    return env
