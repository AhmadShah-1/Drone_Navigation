import gymnasium as gym
from stable_baselines3 import DQN
from drone_env import AirSimDroneEnv
from stable_baselines3.common.env_checker import check_env

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")
check_env(env)

# Define a DQN model for 100 timesteps training
model = DQN(
    'CnnPolicy',
    env,
    learning_rate=0.0006269569008928746,
    buffer_size=100,  # Buffer size to store all experiences
    learning_starts=10,  # Learning starts after 10 steps
    batch_size=10,  # Small batch size
    tau=0.2844690637415575,
    gamma=0.8903399454005295,
    train_freq=1,  # Train after every step
    target_update_interval=10,  # Update target network every 10 steps
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1
)

# Train the model for 100 timesteps
model.learn(total_timesteps=100)

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Optuna_optimized/Models/dqn_airsim_drone_100_steps")
