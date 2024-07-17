import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import gymnasium as gym
from drone_env import AirSimDroneEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
import time

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

# Train the model with a smaller number of timesteps to estimate time
'''
test_timesteps = 1
obs, _ = env.reset()
start_time = time.time()

for _ in range(test_timesteps):
    action, _states = model.predict(obs)
    action = int(action)  # Ensure action is an integer
    obs, reward, done, truncated, info = env.step(action)
    # env.render()  # Call the render method to display the bottom camera feed

    # Print debugging information
    if done:
        print(f"Episode finished due to: {info.get('done_reason')}")
        obs, _ = env.reset()

end_time = time.time()
elapsed_time = end_time - start_time

# Estimate time for 1000 timesteps
estimated_time_1000_timesteps = (elapsed_time / test_timesteps) * 1000
print(f"Time for {test_timesteps} timesteps: {elapsed_time} seconds")
print(f"Estimated time for 1000 timesteps: {estimated_time_1000_timesteps} seconds")
'''

# Train the model for 1000 timesteps
model.learn(total_timesteps=600)

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Models/dqn_airsim_drone2")
