"""NSGA-II 多目标优化模块（基于 pymoo）。

双目标优化:
  f1 = 总行驶距离 (最小化)
  f2 = 完工时间 makespan (最小化)
"""

import time
import random
import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.pm import PM
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV
from assignment import _repair


def _decode_chromosome(x, truck_ids, tasks):
    """将 pymoo 染色体解码为 order 字典。

    Args:
        x: np.ndarray, 每个基因 = 0-based truck index
        truck_ids: 实际卡车ID列表
        tasks: 任务列表

    Returns:
        order: {truck_id: [task_index, ...]}
    """
    order = {tid: [] for tid in truck_ids}
    for task_idx, truck_idx in enumerate(x):
        tid = truck_ids[int(truck_idx)]
        order[tid].append(task_idx)
    return order


def _assignment_from_order(order, tasks):
    """将 order 转换为标准 assignment 格式。"""
    assignment = {tid: [] for tid in order}
    for tid, indices in order.items():
        for idx in indices:
            assignment[tid].append(tasks[idx][0])
    return assignment


def _compute_two_objectives(order, trucks, tasks, dist_cache):
    """计算两个优化目标。

    Returns:
        (total_dist, makespan)
    """
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_start = {t[0]: t[1] for t in trucks}
    speed = trucks[0][2]

    total_dist = 0.0
    truck_times = {}

    for tid in order:
        pos = truck_start[tid]
        dist = 0.0
        for idx in order[tid]:
            jid = tasks[idx][0]
            js, je = task_dict[jid]
            d1 = dist_cache[(pos, js)]
            d2 = dist_cache[(js, je)]
            dist += d1 + d2
            pos = je
        total_dist += dist
        truck_times[tid] = dist / speed

    makespan = max(truck_times.values()) if truck_times else 0.0

    return total_dist, makespan


def _compute_empty_ratio(order, trucks, tasks, dist_cache):
    """仅计算空驶率（用于 Pareto 前沿展示，不作为优化目标）。"""
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_start = {t[0]: t[1] for t in trucks}
    total_dist = 0.0
    total_empty = 0.0
    for tid in order:
        pos = truck_start[tid]
        for idx in order[tid]:
            jid = tasks[idx][0]
            js, je = task_dict[jid]
            d1 = dist_cache[(pos, js)]
            d2 = dist_cache[(js, je)]
            total_empty += d1
            total_dist += d1 + d2
            pos = je
    return total_empty / total_dist if total_dist > 0 else 0.0


class TruckDispatchProblem(ElementwiseProblem):
    """港口集卡调度多目标优化问题（pymoo 封装，双目标）。"""

    def __init__(self, num_tasks, num_trucks, truck_ids, trucks, tasks, dist_cache):
        self.truck_ids = truck_ids
        self.trucks = trucks
        self.tasks = tasks
        self.dist_cache = dist_cache
        self.num_trucks = num_trucks

        super().__init__(
            n_var=num_tasks,
            n_obj=2,
            xl=0,
            xu=num_trucks - 1,
            vtype=int,
        )

    def _evaluate(self, x, out, *args, **kwargs):
        """评估一个个体（染色体）的两个目标。"""
        x_list = x.tolist() if hasattr(x, 'tolist') else list(x)

        # 解码染色体
        order = _decode_chromosome(x_list, self.truck_ids, self.tasks)

        # 修复染色体（确保所有任务都被分配）
        truck_ids = self.truck_ids
        num_tasks = len(self.tasks)
        chrom_copy = list(x_list)
        _repair(chrom_copy, order, num_tasks, truck_ids)

        # 计算两个目标
        total_dist, makespan = _compute_two_objectives(
            order, self.trucks, self.tasks, self.dist_cache
        )

        out["F"] = [total_dist, makespan]


def _precompute_distance_cache(graph, coords, path_fn):
    """预计算所有节点对的最短距离。"""
    node_ids = list(graph.keys())
    cache = {}
    for s in node_ids:
        for e in node_ids:
            if s != e and (s, e) not in cache:
                _, d = path_fn(graph, coords, s, e)
                cache[(s, e)] = d
                cache[(e, s)] = d
        cache[(s, s)] = 0.0
    return cache


def _topsis_select(pareto_front):
    """TOPSIS 多属性决策: 从 Pareto 前沿选最均衡的方案。"""
    if not pareto_front:
        return None
    n = len(pareto_front)
    if n == 1:
        return pareto_front[0]

    dists = np.array([p["total_dist"] for p in pareto_front])
    mss = np.array([p["makespan"] for p in pareto_front])

    # 归一化 (min-max)
    d_min, d_max = dists.min(), dists.max()
    m_min, m_max = mss.min(), mss.max()
    d_range = d_max - d_min if d_max > d_min else 1
    m_range = m_max - m_min if m_max > m_min else 1

    norm_d = (dists - d_min) / d_range
    norm_m = (mss - m_min) / m_range

    # 理想解 (最小化, 所以理想=0, 负理想=1)
    ideal = np.array([0.0, 0.0])
    nadir = np.array([1.0, 1.0])

    # 到理想解和负理想解的距离
    s_plus = np.sqrt((norm_d - ideal[0])**2 + (norm_m - ideal[1])**2)
    s_minus = np.sqrt((norm_d - nadir[0])**2 + (norm_m - nadir[1])**2)

    # 相对贴近度
    c = s_minus / (s_plus + s_minus + 1e-10)
    best_idx = int(np.argmax(c))

    print(f"  TOPSIS 选解: 总距离={pareto_front[best_idx]['total_dist']:.2f}, "
          f"完工时间={pareto_front[best_idx]['makespan']:.2f}, "
          f"贴近度={c[best_idx]:.3f}")
    return pareto_front[best_idx]


def run_nsga2(graph, coords, trucks, tasks, path_fn,
              pop_size=200, n_gen=200, select_by="knee", verbose=True):
    """运行 NSGA-II 多目标优化。

    Args:
        graph: 路网图
        coords: 节点坐标
        trucks: 卡车列表
        tasks: 任务列表
        path_fn: 路径规划函数
        pop_size: 种群大小
        n_gen: 迭代代数
        select_by: 从 Pareto 前沿选择最终方案的标准
                   ("makespan", "total_dist", "knee")
        verbose: 是否打印进度

    Returns:
        assignment: {truck_id: [task_id, ...]}
        runtime: 运行时间（秒）
        pareto_front: [{total_dist, makespan, empty_ratio, assignment}, ...]
        hv_history: list of HV values per generation
    """
    start_time = time.time()

    truck_ids = [t[0] for t in trucks]
    num_tasks = len(tasks)
    num_trucks = len(trucks)

    print(f"预计算距离缓存 ({len(graph)} 节点)...")
    dist_cache = _precompute_distance_cache(graph, coords, path_fn)

    print(f"初始化 NSGA-II (pop={pop_size}, gen={n_gen}, obj=2)...")
    problem = TruckDispatchProblem(
        num_tasks, num_trucks, truck_ids, trucks, tasks, dist_cache
    )

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=IntegerRandomSampling(),
        crossover=TwoPointCrossover(prob=0.9),
        mutation=PM(prob=0.15, eta=20, repair=RoundingRepair()),
        eliminate_duplicates=True,
    )

    print(f"运行多目标优化 (启用 HV 跟踪)...")
    res = minimize(
        problem,
        algorithm,
        ("n_gen", n_gen),
        seed=1,
        verbose=verbose,
        save_history=True,
    )

    # 提取 Pareto 前沿
    pareto_front = []
    for i in range(len(res.F)):
        x = res.X[i]
        x_list = x.tolist() if hasattr(x, 'tolist') else list(x)
        order = _decode_chromosome(x_list, truck_ids, tasks)
        assignment = _assignment_from_order(order, tasks)
        td, ms = _compute_two_objectives(order, trucks, tasks, dist_cache)
        # 计算空驶率（仅用于展示）
        er = _compute_empty_ratio(order, trucks, tasks, dist_cache)
        pareto_front.append({
            "total_dist": float(td),
            "makespan": float(ms),
            "empty_ratio": float(er),
            "assignment": assignment,
        })

    # 按总距离排序
    pareto_front.sort(key=lambda p: p["total_dist"])

    # === HV 收敛曲线 ===
    hv_history = []
    try:
        ref_point = np.array([
            max(p["total_dist"] for p in pareto_front) * 1.1 if pareto_front else 1,
            max(p["makespan"] for p in pareto_front) * 1.1 if pareto_front else 1,
        ])
        hv_indicator = HV(ref_point=ref_point)
        for entry in res.history:
            F = entry.pop.get("F")
            if F is not None and len(F) > 0:
                hv_history.append(hv_indicator.do(F))
    except Exception as e:
        print(f"  ⚠ HV 计算失败: {e}")
        hv_history = []

    # 从 Pareto 前沿选择一个最终方案
    if select_by == "topsis":
        best_p = _topsis_select(pareto_front)
    elif select_by == "knee":
        if len(pareto_front) > 0:
            min_dist = min(p["total_dist"] for p in pareto_front)
            min_ms = min(p["makespan"] for p in pareto_front)
            range_dist = max(p["total_dist"] for p in pareto_front) - min_dist or 1
            range_ms = max(p["makespan"] for p in pareto_front) - min_ms or 1

            def _knee_dist(p):
                return (((p["total_dist"] - min_dist) / range_dist) ** 2 +
                        ((p["makespan"] - min_ms) / range_ms) ** 2)

            best_p = min(pareto_front, key=_knee_dist)
        else:
            best_p = None
    else:
        key = select_by if select_by in ("total_dist", "makespan") else "makespan"
        candidates = sorted(pareto_front, key=lambda p: p[key])
        best_p = candidates[0] if candidates else None

    final_assignment = best_p["assignment"] if best_p else {t: [] for t in truck_ids}

    runtime = time.time() - start_time
    print(f"\nNSGA-II 完成，耗时 {runtime:.2f}s")
    print(f"Pareto 前沿包含 {len(pareto_front)} 个解")
    if hv_history:
        print(f"HV 收敛: {hv_history[0]:.1f} → {hv_history[-1]:.1f}")
    if best_p:
        print(f"选择方案: 总距离={best_p['total_dist']:.2f}, "
              f"完工时间={best_p['makespan']:.2f}, "
              f"空驶率={best_p['empty_ratio']:.3f}")

    return final_assignment, runtime, pareto_front, hv_history
