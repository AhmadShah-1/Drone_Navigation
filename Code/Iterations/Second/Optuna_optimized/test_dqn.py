import numpy as np
import gymnasium as gym
from drone_env import AirSimDroneEnv
from stable_baselines3 import DQN

# Define a function to test the model with target coordinates
def test_model(model, env, target_position, episodes=10):
    for episode in range(episodes):
        obs, _ = env.reset(target_position=target_position)
        done = False
        total_reward = 0

        while not done:
            action, _states = model.predict(obs)
            action = int(action)  # Ensure action is an integer
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            # env.render()  # Display the bottom camera feed if needed

            if done:
                print(f"Episode {episode + 1} finished with total reward: {total_reward}")
                print(f"Done reason: {info.get('done_reason')}")
                break

    env.close()

# Create the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")

# Load the trained model
model_path = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Optuna_optimized/Models/dqn_airsim_drone_100_steps.zip"
model = DQN.load(model_path, env=env)

# Define target coordinates
target_position = np.array([30, 3, -10])  # Example target position

# Test the model with the target coordinates
test_model(model, env, target_position)
