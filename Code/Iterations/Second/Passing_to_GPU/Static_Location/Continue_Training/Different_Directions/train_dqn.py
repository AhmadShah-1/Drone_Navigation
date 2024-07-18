import gymnasium as gym
from stable_baselines3 import DQN
from drone_env import AirSimDroneEnv
from stable_baselines3.common.env_checker import check_env
import numpy as np


# Create and check the custom environment with the new target position

# Backwards
# new_target_position = np.array([-30, -3, -15])

# Right (Fairly Certain, did not test this yet)
new_target_position = np.array([0, 30, -15])

env = AirSimDroneEnv(ip_address="127.0.0.1", device='cuda')
check_env(env)

# Load the existing model
model_path = "/Iterations/Second/Passing_to_GPU/Static_Location/Models/Forward_backward.zip"
model = DQN.load(model_path, env=env)

# Update the environment's target position
env.reset(target_position=new_target_position)

# Continue training the model for additional timesteps
model.learn(total_timesteps=500)

# Save the updated model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Optuna_optimized/Models/Forward_backward_Right")
