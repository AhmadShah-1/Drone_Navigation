import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from drone_env import AirSimDroneEnv
from stable_baselines3 import DQN
import time

# Create the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")

# Load the trained model
model_path = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Models/dqn_airsim_drone2"
model = DQN.load(model_path, env=env)

# Test the model
test_episodes = 10  # Number of test episodes
for episode in range(test_episodes):
    obs, _ = env.reset()
    done = False
    total_reward = 0

    while not done:
        action, _states = model.predict(obs)
        action = int(action)  # Ensure action is an integer
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        env.render()  # Display the bottom camera feed if needed

        if done:
            print(f"Episode {episode + 1} finished with total reward: {total_reward}")
            print(f"Done reason: {info.get('done_reason')}")
            break

env.close()
