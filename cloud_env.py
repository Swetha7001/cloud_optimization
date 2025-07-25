import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CloudLoadBalancerEnv(gym.Env):
    def __init__(self):
        super(CloudLoadBalancerEnv, self).__init__()
        self.num_servers = 3
        self.max_queue = 10
        self.max_steps = 200
        self.current_step = 0

        self.action_space = spaces.Discrete(self.num_servers)
        self.observation_space = spaces.Box(
            low=0,
            high=self.max_queue,
            shape=(self.num_servers,),
            dtype=np.int32
        )

        self.state = np.zeros(self.num_servers, dtype=np.int32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = np.zeros(self.num_servers, dtype=np.int32)
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1

        if self.state[action] < self.max_queue:
            self.state[action] += 1

        self.state = np.maximum(0, self.state - 1)

        reward = 1.0 - (np.std(self.state) / self.max_queue)
        terminated = self.current_step >= self.max_steps
        truncated = False

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Server Loads: {self.state}")

    @property
    def server_loads(self):
        return self.state.copy()
