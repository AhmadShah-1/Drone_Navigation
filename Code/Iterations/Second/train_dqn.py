import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from drone_env import AirSimDroneEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")
check_env(env)

# Define a simple DQN model
model = DQN(
    'CnnPolicy',
    env,
    learning_rate=0.0001,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=32,
    tau=0.1,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    verbose=1
)

# Train the model
model.learn(total_timesteps=1000)

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/First/dqn_airsim_drone1")
