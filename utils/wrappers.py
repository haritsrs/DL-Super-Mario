import gym
import numpy as np
from gym import ObservationWrapper, Wrapper
import cv2


class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(ObservationWrapper):
    def __init__(self, env, keep_dim=False):
        super().__init__(env)
        self.keep_dim = keep_dim
        obs_shape = self.observation_space.shape[:2]
        if self.keep_dim:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(*obs_shape, 1), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        if self.keep_dim:
            gray = np.expand_dims(gray, axis=-1)
        return gray


class ResizeObservation(ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.shape[0], self.shape[1], 1), dtype=np.uint8)

    def observation(self, observation):
        resized = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        return np.expand_dims(resized, axis=-1)


class NormalizeObservation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = env.observation_space

    def observation(self, observation):
        return np.array(observation).astype(np.float32) / 255.0
