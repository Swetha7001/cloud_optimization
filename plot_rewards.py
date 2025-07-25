import numpy as np
import matplotlib.pyplot as plt

# Load the rewards
rewards = np.load("logs/reward_logs.npy")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(rewards, label="Reward per step")
plt.title("Cloud Load Balancer – DQN Reward Trend")
plt.xlabel("Timestep")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("logs/reward_plot.png")  # Optional: save the plot
plt.show()
