import gymnasium as gym
from stable_baselines3 import DQN
from drone_env import AirSimDroneEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np


# Create and check the custom environment with the new target position
new_target_position = np.array([-30, -3, -15])
env = AirSimDroneEnv(ip_address="127.0.0.1", device='cuda')
check_env(env)

# Load the existing model
model_path = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Optuna_optimized/Models/usingGPU3"
model = DQN.load(model_path, env=env)

# Update the environment's target position
env.reset(target_position=new_target_position)

# Continue training the model for additional timesteps
model.learn(total_timesteps=1000)

# Save the updated model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Optuna_optimized/Models/forward_backward")
