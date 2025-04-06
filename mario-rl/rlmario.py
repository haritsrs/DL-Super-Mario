import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import RIGHT_ONLY

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
env = JoypadSpace(env, RIGHT_ONLY)

state = env.reset()
done = False

while not done:
    state, reward, done, info = env.step(1)  # Always press RIGHT

    env.render()
