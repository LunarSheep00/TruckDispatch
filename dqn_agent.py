"""Deep Q-Network (DQN) 智能体用于港口集卡调度。

用神经网络替代 Q-table 处理高维状态空间，支持 Experience Replay 和 Target Network。
"""

import random
import time
import numpy as np
from collections import deque

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = None
    optim = None


class DQN(nn.Module):
    """简单前馈神经网络: 状态 -> 各动作的 Q 值。"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """经验回放缓冲区。"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        state = torch.FloatTensor(np.array([b[0] for b in batch]))
        action = torch.LongTensor(np.array([b[1] for b in batch]))
        reward = torch.FloatTensor(np.array([b[2] for b in batch]))
        next_state = torch.FloatTensor(np.array([b[3] for b in batch]))
        done = torch.FloatTensor(np.array([b[4] for b in batch]))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """Deep Q-Network 智能体。

    用神经网络拟合 Q(s,a)，支持 Experience Replay 与 Target Network。
    """

    def __init__(self, env, learning_rate=1e-3, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.998,
                 batch_size=64, target_update=50, memory_size=20000):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for DQN agent. Install with: pip install torch")

        self.env = env
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        self.num_actions = env.num_trucks
        self.state_dim = self._build_state_dim()

        # 主网络与目标网络
        self.q_net = DQN(self.state_dim, self.num_actions)
        self.target_net = DQN(self.state_dim, self.num_actions)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(capacity=memory_size)
        self.train_step = 0

    def _build_state_dim(self):
        """计算状态向量的维度。"""
        return (self.env.num_trucks            # 每辆卡车所在区域类型
                + 1                             # 当前任务起点类型
                + 1                             # 当前任务终点类型
                + 1                             # 最近卡车距离区间
                + 1                             # 平均负载区间
                + 1)                            # 完成进度

    def _encode_state(self, state_tuple):
        """将环境返回的状态元组编码为连续向量。

        state_tuple: (start_type, end_type, nearest_dist_bin, avg_load_bin)
        """
        start_t, end_t, dist_bin, load_bin = state_tuple

        # 卡车位置编码（从环境中获取）
        pos_regions = []
        for tid in self.env.truck_ids:
            pos = self.env.truck_pos.get(tid, 0)
            from rl_env import _node_type_id as nti
            pos_regions.append(float(nti(pos)))

        # 进度
        progress = self.env.task_idx / max(self.env.num_tasks, 1)

        vec = (pos_regions
               + [float(start_t), float(end_t),
                  float(dist_bin), float(load_bin),
                  progress])
        return np.array(vec, dtype=np.float32)

    def _get_action(self, state_vec):
        """epsilon-greedy 策略选择动作。"""
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        with torch.no_grad():
            q_values = self.q_net(torch.FloatTensor(state_vec).unsqueeze(0))
            return int(q_values.argmax().item())

    def train(self, episodes=5000, verbose=True):
        """训练 DQN 智能体。

        Args:
            episodes: 训练轮数
            verbose: 是否打印进度

        Returns:
            reward_history: 每轮总奖励列表
        """
        reward_history = []

        for ep in range(episodes):
            raw_state = self.env.reset()
            state_vec = self._encode_state(raw_state)
            total_reward = 0.0
            done = False

            while not done:
                action = self._get_action(state_vec)
                next_raw_state, reward, done, _ = self.env.step(action)
                next_state_vec = (self._encode_state(next_raw_state)
                                  if next_raw_state is not None
                                  else np.zeros(self.state_dim, dtype=np.float32))

                self.memory.push(state_vec, action, reward, next_state_vec, done)
                total_reward += reward
                state_vec = next_state_vec

                # 训练
                if len(self.memory) >= self.batch_size:
                    self._train_step()

            reward_history.append(total_reward)

            # 衰减 epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (ep + 1) % 100 == 0:
                print(f"  Episode {ep+1}/{episodes}, Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")

        return reward_history

    def _train_step(self):
        """执行一步梯度更新。"""
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)

        # 当前 Q 值
        q_values = self.q_net(state)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_net(next_state).max(1)[0]
            target = reward + self.gamma * next_q * (1 - done)

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_step += 1
        if self.train_step % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def greedy_assignment(self):
        """使用训练好的策略（贪心，无探索）生成任务分配。"""
        raw_state = self.env.reset()
        state_vec = self._encode_state(raw_state)
        done = False

        while not done:
            with torch.no_grad():
                q_values = self.q_net(torch.FloatTensor(state_vec).unsqueeze(0))
                action = int(q_values.argmax().item())
            next_raw_state, _, done, _ = self.env.step(action)
            if next_raw_state is not None:
                state_vec = self._encode_state(next_raw_state)

        return self.env.get_assignment()


def run_dqn(graph, coords, trucks, tasks, path_fn=None, episodes=5000):
    """一站式运行 DQN 调度。

    Args:
        graph: 路网图
        coords: 节点坐标
        trucks: 卡车列表
        tasks: 动态任务列表（含 arrival_time）
        path_fn: 路径规划函数（未使用，兼容接口）
        episodes: 训练轮数

    Returns:
        assignment: {truck_id: [task_id, ...]}
        reward_history: 每轮总奖励列表
        runtime: 运行时间（秒）
    """
    from rl_env import PortTruckEnv

    start_time = time.time()

    print(f"初始化港口环境...")
    from pathfinding import astar
    env = PortTruckEnv(graph, coords, trucks, tasks, astar)

    print(f"训练 DQN (episodes={episodes}, state_dim={env.num_trucks + 4}, actions={env.num_trucks})...")
    agent = DQNAgent(env)
    reward_history = agent.train(episodes=episodes, verbose=True)

    print(f"生成最终分配方案...")
    assignment = agent.greedy_assignment()

    runtime = time.time() - start_time
    print(f"DQN 完成，耗时 {runtime:.2f}s")
    print(f"  Reward: {reward_history[0]:.2f} -> {reward_history[-1]:.2f}")

    return assignment, reward_history, runtime
