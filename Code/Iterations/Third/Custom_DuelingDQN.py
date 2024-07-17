import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from drone_env import AirSimDroneEnv
import time


def make_env(ip_address, drone_names):
    def _init():
        env = AirSimDroneEnv(ip_address=ip_address, drone_names=drone_names)
        return env

    return _init


# Define the number of drones
num_drones = 2
drone_names = [f"Drone{i + 1}" for i in range(num_drones)]

# Create a vectorized environment
env = DummyVecEnv([make_env("127.0.0.1", drone_names)])

# Define the DQN model with dueling network
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

# Train the model for 600 timesteps
model.learn(total_timesteps=600)

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Third/Models/dqn_airsim_drone_multi")
