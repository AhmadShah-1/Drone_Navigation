import os
import time
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
import airsim0
import numpy as np
from gymnasium import spaces
from PIL import Image
import cv2
import torch

# Import the custom environment from your environment file
from drone_env import AirSimDroneEnv

# Define the custom target coordinates
list_of_targets = [
    [0, 30, -15], [30, 0, -15], [-30, 0, -15],
    [0, -30, -15], [3, -30, -15], [-3, -30, -15],
    [-30, 3, -15], [-30, -3, -15]
]

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1", target_positions=list_of_targets, device='cuda')
check_env(env)

# Wrap the environment for Stable Baselines
env = DummyVecEnv([lambda: Monitor(env)])
env = VecTransposeImage(env)

# Load the trained model
model_path = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Models/dqn_airsim_drone_policy"  # Replace with the path to your trained model
model = DQN.load(model_path)

# Test the model
obs = env.reset()
for i in range(100):  # Run for 1000 steps
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, info = env.step(action)
    # env.render()  # Render the environment to visualize

    if dones:
        obs = env.reset()  # Reset environment if done

env.close()

# To view TensorBoard logs, you can run the following command in your terminal:
# tensorboard --logdir=path_to_your_tensorboard_log_directory
