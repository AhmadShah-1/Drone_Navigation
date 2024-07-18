import gymnasium as gym
from stable_baselines3 import DQN
from drone_env import AirSimDroneEnv
from stable_baselines3.common.env_checker import check_env

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")
check_env(env)

# Define a simple DQN model
model = DQN(
    'CnnPolicy',
    env,
    learning_rate=0.01,
    buffer_size=10000,
    learning_starts=1000,
    batch_size=64,
    tau=0.1,
    gamma=0.3,
    train_freq=4,
    target_update_interval=50,
    exploration_fraction=0.1,
    exploration_final_eps=0.05,
    verbose=1
)

# Train the model for 1000 timesteps (you can increase this number for actual training)
model.learn(total_timesteps=3800)

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Models/dqn_airsim_drone3")
