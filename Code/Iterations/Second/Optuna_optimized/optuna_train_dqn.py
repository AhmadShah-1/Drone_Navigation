import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from drone_env import AirSimDroneEnv
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler

# Create and check the custom environment
env = AirSimDroneEnv(ip_address="127.0.0.1")
check_env(env)


# 1 time-steps, 20 trials:
# Best hyperparameters:  {'learning_rate': 0.00010781847115130577, 'buffer_size': 47830, 'learning_starts': 3263, 'batch_size': 32, 'tau': 0.25707732981149245, 'gamma': 0.9146397267081794, 'train_freq': 4, 'target_update_interval': 74, 'exploration_fraction': 0.2451370310701446, 'exploration_final_eps': 0.08274214536830939}

# 10 time-steps, 10 trials:
# Best hyperparameters:  {'learning_rate': 0.0006269569008928746, 'buffer_size': 35316, 'learning_starts': 4369, 'batch_size': 256, 'tau': 0.2844690637415575, 'gamma': 0.8903399454005295, 'train_freq': 7, 'target_update_interval': 76, 'exploration_fraction': 0.30199659573481286, 'exploration_final_eps': 0.04483019360432404}


# Define the objective function for Optuna
def objective(trial: Trial):
    # Define the hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    buffer_size = trial.suggest_int('buffer_size', 10000, 50000)
    learning_starts = trial.suggest_int('learning_starts', 1000, 5000)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    tau = trial.suggest_uniform('tau', 0.05, 0.3)
    gamma = trial.suggest_uniform('gamma', 0.8, 0.99)
    train_freq = trial.suggest_int('train_freq', 1, 8)
    target_update_interval = trial.suggest_int('target_update_interval', 10, 100)
    exploration_fraction = trial.suggest_uniform('exploration_fraction', 0.1, 0.5)
    exploration_final_eps = trial.suggest_uniform('exploration_final_eps', 0.01, 0.1)

    # Define the DQN model with trial hyperparameters
    model = DQN(
        'CnnPolicy',
        env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_final_eps=exploration_final_eps,
        verbose=0
    )

    # Use an evaluation callback to stop training when the reward reaches a certain threshold
    eval_env = AirSimDroneEnv(ip_address="127.0.0.1")
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=500,
                                 deterministic=True, render=False)

    # Train the model
    model.learn(total_timesteps=10, callback=eval_callback)

    # Get the mean reward
    mean_reward = eval_callback.last_mean_reward

    return mean_reward


# Optimize the hyperparameters
study = optuna.create_study(direction='maximize', sampler=TPESampler())
study.optimize(objective, n_trials=10, n_jobs=1)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)



# Save the best model
best_model_params = study.best_params
model = DQN(
    'CnnPolicy',
    env,
    learning_rate=best_model_params['learning_rate'],
    buffer_size=best_model_params['buffer_size'],
    learning_starts=best_model_params['learning_starts'],
    batch_size=best_model_params['batch_size'],
    tau=best_model_params['tau'],
    gamma=best_model_params['gamma'],
    train_freq=best_model_params['train_freq'],
    target_update_interval=best_model_params['target_update_interval'],
    exploration_fraction=best_model_params['exploration_fraction'],
    exploration_final_eps=best_model_params['exploration_final_eps'],
    verbose=1
)

model.learn(total_timesteps=10)
model.save(
    "C:/Users/alex1/Desktop/Ahmad_Stuff/Drone_Navigation/Code/Iterations/Second/Models/dqn_airsim_drone_optimized")
