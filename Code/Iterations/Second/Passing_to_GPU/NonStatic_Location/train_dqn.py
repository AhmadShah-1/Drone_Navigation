import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from drone_env import AirSimDroneEnv
import os

# Define a list of possible target locations
list_of_targets = [[0, 30, -15], [30, 0, -15], [-30, 0, -15], [0, -30, -15], [3, -30, -15], [-3, -30, -15], [-30, 3, -15], [-30, -3, -15]]

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1", target_positions=list_of_targets, device='cuda')
check_env(env)

# Set up TensorBoard log directory
tensorboard_log_dir = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Log/FirstIteration"

# Ensure the log directory exists
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Define a DQN model with adjusted hyperparameters for short-term training and TensorBoard logging
model = DQN(
    'CnnPolicy',
    env,
    learning_rate=0.0001,
    buffer_size=10000,  # Increase buffer size to store more experiences
    learning_starts=50,  # Start learning after 50 steps
    batch_size=32,  # Moderate batch size
    tau=0.1,
    gamma=0.99,
    train_freq=4,  # Train every 4 steps
    target_update_interval=100,  # Update target network every 100 steps
    exploration_fraction=0.5,
    exploration_final_eps=0.1,
    verbose=1,
    tensorboard_log=tensorboard_log_dir  # Enable TensorBoard logging
)

# Train the model for 500 timesteps (ensure sufficient duration to generate logs)
model.learn(total_timesteps=1300, tb_log_name="DQN_airsim_drone")

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Models/FirstIteration")

# To view TensorBoard logs, you can run the following command in your terminal:
# tensorboard --logdir=C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Log/FirstIteration/
