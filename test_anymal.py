import gymnasium as gym

env = gym.make("AnymalC-v0", render_mode="human")
env.reset()

while True:
    obs, rew, terminated, truncated, info = env.step(env.action_space.sample())
    if terminated or truncated:
        env.reset()
