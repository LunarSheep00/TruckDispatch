"""Tabular Q-Learning 智能体用于港口集卡调度。"""

import random
import time
import numpy as np
from rl_env import PortTruckEnv
from data import get_graph, get_coords, get_trucks, get_dynamic_tasks
from pathfinding import astar


class QLearningAgent:
    """Tabular Q-Learning 智能体。

    使用 Q-table 存储状态-动作值，epsilon-greedy 策略探索。
    """

    def __init__(self, env, learning_rate=0.3, discount_factor=0.95,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.999):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.num_actions = env.num_trucks
        self.q_table = {}  # state -> np.array of q-values

    def _get_q_values(self, state):
        """获取状态的 Q 值向量，如不存在则初始化为零。"""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)
        return self.q_table[state]

    def _get_action(self, state):
        """epsilon-greedy 策略选择动作。"""
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        q_values = self._get_q_values(state)
        # 如果有多个相同最大值，随机选一个
        max_q = np.max(q_values)
        candidates = np.where(q_values == max_q)[0]
        return int(random.choice(candidates))

    def _update_q(self, state, action, reward, next_state):
        """Q-learning 更新规则:
        Q(s,a) += lr * (reward + gamma * max(Q(s',a')) - Q(s,a))
        """
        q_values = self._get_q_values(state)
        if next_state is not None:
            next_q = self._get_q_values(next_state)
            max_next = np.max(next_q)
        else:
            max_next = 0.0

        td_target = reward + self.gamma * max_next
        td_error = td_target - q_values[action]
        q_values[action] += self.lr * td_error

    def train(self, episodes=5000, verbose=True):
        """训练 Q-Learning 智能体。

        Args:
            episodes: 训练轮数
            verbose: 是否打印进度

        Returns:
            reward_history: 每轮的总奖励列表
        """
        reward_history = []

        for ep in range(episodes):
            state = self.env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self._get_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self._update_q(state, action, reward, next_state)
                total_reward += reward
                state = next_state

            reward_history.append(total_reward)

            # 衰减 epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            if verbose and (ep + 1) % 50 == 0:
                print(f"  Episode {ep+1}/{episodes}, Total Reward: {total_reward:.2f}, "
                      f"Epsilon: {self.epsilon:.3f}")

        return reward_history

    def greedy_assignment(self):
        """使用训练好的策略（贪心，无探索）生成任务分配。"""
        state = self.env.reset()
        done = False

        while not done:
            q_values = self._get_q_values(state)
            action = int(np.argmax(q_values))
            next_state, _, done, _ = self.env.step(action)
            state = next_state

        return self.env.get_assignment()


def run_q_learning(graph, coords, trucks, tasks, path_fn=astar, episodes=5000):
    """一站式运行 Q-Learning 调度。

    Args:
        graph: 路网图
        coords: 节点坐标
        trucks: 卡车列表
        tasks: 动态任务列表（需包含 arrival_time）
        path_fn: 路径规划函数
        episodes: 训练轮数

    Returns:
        assignment: {truck_id: [task_id, ...]}
        reward_history: 每轮总奖励列表
        runtime: 运行时间（秒）
    """
    start_time = time.time()

    print(f"初始化港口环境...")
    env = PortTruckEnv(graph, coords, trucks, tasks, path_fn)

    print(f"训练 Q-Learning (episodes={episodes})...")
    agent = QLearningAgent(env)
    reward_history = agent.train(episodes=episodes, verbose=True)

    print(f"生成最终分配方案...")
    assignment = agent.greedy_assignment()

    runtime = time.time() - start_time
    print(f"Q-Learning 完成，耗时 {runtime:.2f}s")

    return assignment, reward_history, runtime
