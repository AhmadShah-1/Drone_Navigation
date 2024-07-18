import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_env import AirSimDroneEnv
import time

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")
check_env(env)

# Load the existing model
model_path = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Third/Models/dqn_airsim_drone_multi"
model = DQN.load(model_path, env=env)

# Continue training the model for additional timesteps
additional_timesteps = 600
model.learn(total_timesteps=additional_timesteps)

# Save the model again after additional training
model.save(model_path)
