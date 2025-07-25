from agents.dqn_agent import load_trained_model
from env.cloud_env import CloudLoadBalancerEnv
import time
import gym
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = load_trained_model()
    env = CloudLoadBalancerEnv()
    obs, _ = env.reset()

    done = False
    total_reward = 0
    step_count = 0

    print("\n🤖 Testing the trained AI Load Balancer with Visualization...\n")

    plt.ion()
    fig, ax = plt.subplots()
    bars = ax.bar(range(env.num_servers), env.server_loads)
    ax.set_ylim(0, env.max_queue)
    ax.set_xlabel("Server")
    ax.set_ylabel("Load")
    ax.set_title("AI Load Balancer Server Loads")

    while not done and step_count < 50:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
        total_reward += reward
        step_count += 1

        # Update the plot
        for i, bar in enumerate(bars):
            bar.set_height(env.server_loads[i])
        plt.pause(0.1)

    print(f"\n🎯 Total Steps Taken: {step_count}")
    print(f"🏆 Total Reward: {total_reward:.2f}")
    print("✅ AI agent has successfully completed the test run.")
    plt.ioff()
    plt.show()