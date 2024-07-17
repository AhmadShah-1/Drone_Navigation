import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import time

# Import the custom environment
from drone_env import AirSimDroneEnv

def make_env(ip_address, rank):
    """
    Utility function to create a multiprocess environment
    """
    def _init():
        env = AirSimDroneEnv(ip_address=f"{ip_address}:{41451 + rank}")  # Assuming each drone can be accessed via a different port
        return env
    return _init

if __name__ == "__main__":
    # Number of parallel environments
    num_envs = 4

    # Create the vectorized environment
    env = SubprocVecEnv([make_env("127.0.0.1", i) for i in range(num_envs)])

    # Check the environment
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
        gamma=0.2,       # Lower focuses on short-term rewards, higher focuses on long-term rewards
        train_freq=4,
        target_update_interval=2,
        exploration_fraction=0.3,
        exploration_final_eps=0.05,
        verbose=1
    )

    # Train the model for 1000 timesteps
    model.learn(total_timesteps=1000)

    # Save the model
    model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Models/dqn_airsim_drone2")
