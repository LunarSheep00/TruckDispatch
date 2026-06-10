"""港口集卡调度强化学习环境 (Port Truck Dispatch Environment for Q-Learning)。"""

from collections import defaultdict
from pathfinding import astar

_BERTHS = {1,2,3,4,5,8,9,12}
_GATES  = {38,39}

def _node_type_id(nid):
    """将节点 ID 映射为功能类型（3类），大幅压缩状态空间。"""
    if nid in _BERTHS:
        return 0  # berth
    if nid in _GATES:
        return 1  # gate
    return 2      # warehouse


class PortTruckEnv:
    """港口卡车调度环境。

    状态: (task_start, task_end, nearest_truck_dist_bin, avg_truck_load_bin)
    动作: truck_id (映射到实际卡车ID)
    奖励: -distance - 0.5*waiting_time - 0.3*congestion_penalty
    """

    def __init__(self, graph, coords, trucks, tasks, path_fn=astar):
        self.graph = graph
        self.coords = coords
        self.trucks = trucks
        self.tasks = sorted(tasks, key=lambda t: t[3])  # 按到达时间排序
        self.path_fn = path_fn
        self.speed = trucks[0][2]  # km/h

        self.truck_ids = [t[0] for t in trucks]
        self.num_trucks = len(self.truck_ids)
        self.num_tasks = len(self.tasks)

        # 预计算距离缓存
        self._build_dist_cache()

        self.reset()

    def _build_dist_cache(self):
        """预计算所有节点对之间的最短距离。"""
        node_ids = list(self.graph.keys())
        self.dist_cache = {}
        for s in node_ids:
            for e in node_ids:
                if s != e and (s, e) not in self.dist_cache:
                    _, d = self.path_fn(self.graph, self.coords, s, e)
                    self.dist_cache[(s, e)] = d
                    self.dist_cache[(e, s)] = d
            self.dist_cache[(s, s)] = 0.0

    def reset(self):
        """重置环境到初始状态。"""
        self.truck_pos = {t[0]: t[1] for t in self.trucks}
        self.truck_tasks = {t[0]: [] for t in self.trucks}  # 已分配任务ID列表
        self.truck_paths = {t[0]: [t[1]] for t in self.trucks}
        self.truck_completed = {t[0]: False for t in self.trucks}

        self.current_time = 0.0  # 当前仿真时间（分钟）
        self.task_idx = 0  # 下一个要分配的任务索引
        self.edge_usage = defaultdict(int)
        self.total_distance = 0.0
        self.total_empty_dist = 0.0

        # 跟踪每辆卡车的最后位置（用于空驶计算）
        self.truck_last_pos = {t[0]: t[1] for t in self.trucks}

        self.done = False

        # 用于增量拥堵惩罚
        self._prev_congestion = 0.0

        return self._get_state()

    def _discretize_dist(self, d):
        """将距离离散化为5个区间。"""
        if d == 0:
            return 0
        elif d < 5:
            return 1
        elif d < 10:
            return 2
        elif d < 20:
            return 3
        else:
            return 4

    def _discretize_load(self, n_tasks):
        """将任务数离散化为4个区间。"""
        if n_tasks == 0:
            return 0
        elif n_tasks <= 3:
            return 1
        elif n_tasks <= 6:
            return 2
        else:
            return 3

    def _get_state(self):
        """构建当前状态元组。"""
        if self.task_idx >= self.num_tasks:
            return None

        tid, js, je, arrival = self.tasks[self.task_idx]

        # 找到最近的空闲卡车
        min_dist = float('inf')
        for t_id in self.truck_ids:
            if not self.truck_completed[t_id]:
                d = self.dist_cache.get((self.truck_pos[t_id], js), float('inf'))
                if d < min_dist:
                    min_dist = d

        # 平均负载
        loads = [len(self.truck_tasks[tid]) for tid in self.truck_ids]
        avg_load = sum(loads) / self.num_trucks if self.num_trucks > 0 else 0

        return (
            _node_type_id(js),                 # 起点类型：0=berth 1=gate 2=warehouse
            _node_type_id(je),                 # 终点类型
            self._discretize_dist(min_dist),   # 最近卡车距离区间
            self._discretize_load(int(avg_load)),  # 平均负载区间
        )

    def _compute_edge_usage_penalty(self):
        """计算当前边使用量的拥堵惩罚。"""
        if not self.edge_usage:
            return 0.0
        max_usage = max(self.edge_usage.values())
        if max_usage == 0:
            return 0.0
        threshold = max_usage * 0.7
        penalty = 0.0
        for u in self.edge_usage.values():
            if u > threshold:
                penalty += (u - threshold) ** 2
        return penalty

    def step(self, action):
        """执行动作：将当前任务分配给指定卡车。"""
        if self.done:
            return self._get_state(), 0.0, True, {}

        if self.task_idx >= self.num_tasks:
            self.done = True
            return None, 0.0, True, {}

        jid, js, je, arrival = self.tasks[self.task_idx]
        truck_id = self.truck_ids[action]

        # 等待时间：如果任务到达时卡车还没准备好
        waiting = max(0, arrival - self.current_time)
        self.current_time = max(self.current_time, arrival)

        # 计算路径距离
        pos = self.truck_pos[truck_id]
        empty_dist = self.dist_cache.get((pos, js), 0)
        load_dist = self.dist_cache.get((js, je), 0)
        total_trip_dist = empty_dist + load_dist

        # 更新卡车位置
        self.truck_pos[truck_id] = je
        self.truck_tasks[truck_id].append(jid)

        # 记录路径
        path1, _ = self.path_fn(self.graph, self.coords, pos, js)
        path2, _ = self.path_fn(self.graph, self.coords, js, je)
        full_path = path1[:-1] + path2 if path2 else path1
        self.truck_paths[truck_id].extend(full_path[1:])

        # 统计边使用（用于拥堵可视化）
        for p in [path1, path2]:
            for i in range(len(p) - 1):
                u, v = p[i], p[i + 1]
                self.edge_usage[(min(u, v), max(u, v))] += 1

        # 累加距离
        self.total_distance += total_trip_dist
        self.total_empty_dist += empty_dist

        # 更新时间（行驶时间 = 距离/速度，转换为分钟）
        travel_time = (total_trip_dist / self.speed) * 60
        self.current_time += travel_time

        # ========== 奖励函数 v3 (delta congestion) ==========
        # 关键：拥堵惩罚用增量。每步奖励范围 ~ -2 到 +3

        # 拥堵增量惩罚（避免累计值爆炸）
        current_congestion = self._compute_edge_usage_penalty()
        delta_congestion = current_congestion - self._prev_congestion
        self._prev_congestion = current_congestion

        # 负载均衡
        loads = [len(self.truck_tasks[tid]) for tid in self.truck_ids]
        max_load = max(loads) if loads else 0
        avg_load = sum(loads) / len(loads) if loads else 0
        load_imbalance = (max_load - avg_load) / (avg_load + 1)

        base_reward = 2.0                          # +2.0
        dist_penalty = -total_trip_dist / 30       # ~ -0.2 ~ -1.3
        wait_penalty = -0.3 * max(0, waiting) / 10 # ~ 0 ~ -0.3
        cong_penalty = -0.1 * delta_congestion     # ~ 0 ~ -1.0 (delta)
        load_penalty = -0.3 * load_imbalance        # ~ 0 ~ -0.6

        # 终局奖励 +5
        all_done = (self.task_idx + 1) >= self.num_tasks
        completion_bonus = 5.0 if all_done else 0.0

        reward = (base_reward + dist_penalty + wait_penalty + cong_penalty
                  + load_penalty + completion_bonus)

        # 移动到下一个任务
        self.task_idx += 1

        # 检查是否完成
        if self.task_idx >= self.num_tasks:
            self.done = True
            return None, reward, True, {}

        next_state = self._get_state()
        return next_state, reward, False, {}

    def get_assignment(self):
        """返回标准格式的分配结果 {truck_id: [task_id, ...]}。"""
        return {tid: list(tasks) for tid, tasks in self.truck_tasks.items()}

    def get_truck_paths(self):
        """返回每辆卡车的路径 {truck_id: [node, ...]}。"""
        return dict(self.truck_paths)

    def get_edge_usage(self):
        """返回边使用统计。"""
        return dict(self.edge_usage)
