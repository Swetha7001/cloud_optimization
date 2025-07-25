import sys
import os
import numpy as np

# Fix path to access parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from env.cloud_env import CloudLoadBalancerEnv
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy

def train_agent(save_path="models/dqn_cloud"):
    env = CloudLoadBalancerEnv()
    check_env(env)

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=100,
        batch_size=32,
        gamma=0.99,
        target_update_interval=500,
        train_freq=1,
        exploration_fraction=0.1,
        exploration_final_eps=0.01,
        tensorboard_log="./logs"
    )

    model.learn(total_timesteps=10000)
    
    os.makedirs(save_path, exist_ok=True)
    model.save(os.path.join(save_path, "dqn_model"))
    print(f"✅ Model saved to {save_path}/dqn_model")

    return model, env

def load_trained_model(path="models/dqn_cloud/dqn_model"):
    env = CloudLoadBalancerEnv()
    model = DQN.load(path, env=env)
    return model

def evaluate_trained_model(model, env, num_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=num_episodes, render=False)
    print(f"📊 Evaluation over {num_episodes} episodes: Mean Reward = {mean_reward:.2f}, Std = {std_reward:.2f}")

def collect_rewards(model, env, path="logs/reward_logs.npy"):
    reward_list = []
    obs, _ = env.reset()
    for _ in range(200):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        reward_list.append(reward)
        if done:
            break
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure logs directory exists
    np.save(path, reward_list)
    print(f"📈 Reward log saved to {path}")

if __name__ == "__main__":
    model, env = train_agent()
    evaluate_trained_model(model, env)
    collect_rewards(model, env, path="logs/reward_logs.npy")
