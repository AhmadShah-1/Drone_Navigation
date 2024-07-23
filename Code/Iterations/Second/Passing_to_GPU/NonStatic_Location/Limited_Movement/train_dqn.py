import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
import os
import time
from drone_env import AirSimDroneEnv

# Define a list of possible target locations
list_of_targets = [
    [0, 30, -15], [30, 0, -15], [-30, 0, -15],
    [0, -30, -15], [3, -30, -15], [-3, -30, -15],
    [-30, 3, -15], [-30, -3, -15]
]

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1", target_positions=list_of_targets, device='cuda')
check_env(env)

# Wrap the environment for Stable Baselines
env = DummyVecEnv([lambda: Monitor(env)])
env = VecTransposeImage(env)

# Set up TensorBoard log directory
tensorboard_log_dir = "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Log/FirstIteration"
os.makedirs(tensorboard_log_dir, exist_ok=True)

# Define the DQN model with appropriate hyperparameters
model = DQN(
    "CnnPolicy",
    env,
    learning_rate=0.025,
    verbose=1,
    batch_size=16,  # Reduced batch size to decrease memory usage
    train_freq=4,
    target_update_interval=1000,  # Adjusted for shorter training
    learning_starts=1000,  # Adjusted for shorter training
    buffer_size=10000,  # Reduced buffer size to decrease memory usage
    max_grad_norm=10,
    exploration_fraction=0.9,   # Start with higher exploration and slowly bring it down   100 - 50
    exploration_final_eps=0.5,
    device="cuda",
    tensorboard_log=tensorboard_log_dir
)

# Create an evaluation callback
callbacks = []
eval_callback = EvalCallback(
    env,
    callback_on_new_best=None,
    n_eval_episodes=5,
    best_model_save_path="C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Limited_Movement/Models",
    log_path=tensorboard_log_dir,
    eval_freq=1000,  # Adjusted for shorter training
)
callbacks.append(eval_callback)

# Train the model for the specified number of timesteps
model.learn(
    total_timesteps=4000,
    tb_log_name="dqn_airsim_drone_run_" + str(time.time()),
    callback=callbacks
)

# Save the model
model.save("C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Limited_Movement/Models/DQN1")

# To view TensorBoard logs, you can run the following command in your terminal:
# tensorboard --logdir=C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Passing_to_GPU/NonStatic_Location/Log/FirstIteration/
