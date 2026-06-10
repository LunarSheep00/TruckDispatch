import time
import random
from collections import defaultdict

# ----- helpers for congestion-aware metrics -----

def compute_edge_usage(assignment, order, trucks, tasks, dist_cache_astar):
    """统计每条边的使用次数，用于拥堵指数计算。"""
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_start = {t[0]: t[1] for t in trucks}
    usage = defaultdict(int)
    for tid in order:
        pos = truck_start[tid]
        for idx in order[tid]:
            jid = tasks[idx][0]
            js, je = task_dict[jid]
            # 从 pos 到任务起点的路径
            if (pos, js) in dist_cache_astar:
                path1 = dist_cache_astar[(pos, js)]
            if (js, je) in dist_cache_astar:
                path2 = dist_cache_astar[(js, je)]
            for i in range(len(path1) - 1):
                usage[(min(path1[i], path1[i+1]), max(path1[i], path1[i+1]))] += 1
            for i in range(len(path2) - 1):
                usage[(min(path2[i], path2[i+1]), max(path2[i], path2[i+1]))] += 1
            pos = je
    return usage

def _compute_congestion_index(edge_usage):
    """计算拥堵指数：超过阈值的边产生平方惩罚。"""
    if not edge_usage:
        return 0.0
    max_usage = max(edge_usage.values())
    if max_usage == 0:
        return 0.0
    threshold = max_usage * 0.7
    penalty = 0.0
    for usage in edge_usage.values():
        if usage > threshold:
            penalty += (usage - threshold) ** 2
    return penalty

def run_greedy(graph, trucks, tasks, path_fn):
    start_time = time.time()
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_pos = {t[0]: t[1] for t in trucks}
    truck_time = {t[0]: 0.0 for t in trucks}
    assignment = {t[0]: [] for t in trucks}

    for jid, jstart, jend in tasks:
        best_truck = None
        best_dist = float('inf')
        for tid in truck_pos:
            _, d = path_fn(graph, truck_pos[tid], jstart)
            total = truck_time[tid] + d
            if total < best_dist:
                best_dist = total
                best_truck = tid
        _, travel_to_end = path_fn(graph, jstart, jend)
        assignment[best_truck].append(jid)
        truck_pos[best_truck] = jend
        truck_time[best_truck] += best_dist - truck_time[best_truck] + travel_to_end

    return assignment, time.time() - start_time

def _compute_fitness(chromosome, order, trucks, tasks, dist_cache):
    """计算个体适应度。返回 (total_dist, makespan, fitness) 三元组。"""
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
            jstart, jend = task_dict[jid]
            dist += dist_cache[(pos, jstart)] + dist_cache[(jstart, jend)]
            pos = jend
        total_dist += dist
        truck_times[tid] = dist / speed
    makespan = max(truck_times.values()) if truck_times else 0.0
    # 返回原始值，fitness 在主循环中标准化后计算
    return total_dist, makespan, 1.0 / (total_dist + makespan + 1e-4)


def _tournament_select(population, fitnesses, k=3):
    """锦标赛选择：随机选 k 个个体，返回最优的。"""
    best_idx = None
    best_fit = -float('inf')
    for _ in range(k):
        idx = random.randrange(len(population))
        if fitnesses[idx] > best_fit:
            best_fit = fitnesses[idx]
            best_idx = idx
    return population[best_idx]


# ----- CA-GA: congestion-aware fitness -----

def _compute_ca_fitness(chromosome, order, trucks, tasks, dist_cache, path_cache):
    """拥堵感知适应度函数。返回 (total_dist, makespan, congestion, fitness) 元组。"""
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_start = {t[0]: t[1] for t in trucks}
    speed = trucks[0][2]

    total_dist = 0.0
    total_empty_dist = 0.0
    truck_times = {}

    edge_usage = defaultdict(int)

    for tid in order:
        pos = truck_start[tid]
        dist = 0.0
        empty_dist = 0.0
        for idx in order[tid]:
            jid = tasks[idx][0]
            js, je = task_dict[jid]
            d1 = dist_cache[(pos, js)]
            d2 = dist_cache[(js, je)]
            empty_dist += d1
            dist += d1 + d2
            if (pos, js) in path_cache:
                p = path_cache[(pos, js)]
                for i in range(len(p)-1):
                    edge_usage[(min(p[i], p[i+1]), max(p[i], p[i+1]))] += 1
            if (js, je) in path_cache:
                p = path_cache[(js, je)]
                for i in range(len(p)-1):
                    edge_usage[(min(p[i], p[i+1]), max(p[i], p[i+1]))] += 1
            pos = je
        total_dist += dist
        total_empty_dist += empty_dist
        truck_times[tid] = dist / speed

    makespan = max(truck_times.values()) if truck_times else 0.0

    max_usage = max(edge_usage.values()) if edge_usage else 0
    congestion = 0.0
    if max_usage > 0:
        threshold = max_usage * 0.7
        for u in edge_usage.values():
            if u > threshold:
                congestion += (u - threshold) ** 2

    empty_ratio = total_empty_dist / total_dist if total_dist > 0 else 0.0

    cost = (0.4 * total_dist +
            0.3 * makespan +
            0.2 * congestion +
            0.1 * empty_ratio)
    return total_dist, makespan, congestion, 1.0 / (cost + 1e-4)

def _repair(chromosome, order, num_tasks, truck_ids):
    in_order = set()
    for tid in truck_ids:
        in_order.update(order[tid])
    missing = set(range(num_tasks)) - in_order
    for idx in missing:
        tid = random.choice(truck_ids)
        chromosome[idx] = tid
        order[tid].append(idx)
    seen = set()
    for tid in truck_ids:
        deduped = []
        for idx in order[tid]:
            if idx not in seen:
                seen.add(idx)
                deduped.append(idx)
        order[tid] = deduped

def _get_truck_paths(chromosome, order, graph, coords, trucks, tasks, path_fn):
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_start = {t[0]: t[1] for t in trucks}
    truck_paths = {}
    for tid in {t[0] for t in trucks}:
        pos = truck_start[tid]
        full_path = [pos]
        for idx in order[tid]:
            jid = tasks[idx][0]
            jstart, jend = task_dict[jid]
            seg1, _ = path_fn(graph, coords, pos, jstart)
            seg2, _ = path_fn(graph, coords, jstart, jend)
            full_path += seg1[1:] + seg2[1:]
            pos = jend
        truck_paths[tid] = full_path
    return truck_paths


def _two_opt_improve(order, tasks, dist_cache):
    """2-opt 局部搜索: 反转某卡车的任务段，仅当空驶距离减少时接受。"""
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    improved = False
    for tid in list(order.keys()):
        seq = order[tid]
        if len(seq) < 3:
            continue
        i = random.randint(0, len(seq) - 2)
        j = random.randint(i + 1, len(seq) - 1)

        # 计算边界任务的起终点
        def task_loc(idx):
            """返回 (js, je) 元组"""
            return task_dict[tasks[idx][0]]

        # 反转段 [i, j] 影响的空驶距离:
        #   前驱: t_{i-1} 的终点 → t_i 的起点
        #   后继: t_j 的终点 → t_{j+1} 的起点
        # 反转后变为:
        #   前驱: t_{i-1} 的终点 → t_j 的起点
        #   后继: t_i 的终点 → t_{j+1} 的起点

        _, je_prev = task_loc(seq[i-1]) if i > 0 else (None, None)
        js_i, je_i = task_loc(seq[i])
        js_j, je_j = task_loc(seq[j])
        js_next, _ = task_loc(seq[j+1]) if j < len(seq)-1 else (None, None)

        old_empty = (dist_cache.get((je_prev, js_i), 0) if i > 0 else 0) + \
                    (dist_cache.get((je_j, js_next), 0) if j < len(seq)-1 else 0)
        new_empty = (dist_cache.get((je_prev, js_j), 0) if i > 0 else 0) + \
                    (dist_cache.get((je_i, js_next), 0) if j < len(seq)-1 else 0)

        if new_empty < old_empty:
            new_seq = seq[:i] + list(reversed(seq[i:j+1])) + seq[j+1:]
            order[tid] = new_seq
            improved = True
            break
    return improved

def run_ga(graph, coords, trucks, tasks, path_fn):
    import time
    start_time = time.time()
    truck_ids = [t[0] for t in trucks]
    num_tasks = len(tasks)
    POP = 200
    ITERS = 300
    PC_MAX, PC_MIN = 0.85, 0.5
    PM_MAX, PM_MIN = 0.2, 0.05

    # pre-compute all-pairs shortest distances
    node_ids = list(graph.keys())
    dist_cache = {}
    for s in node_ids:
        for e in node_ids:
            if s != e and (s, e) not in dist_cache:
                _, d = path_fn(graph, coords, s, e)
                dist_cache[(s, e)] = d
                dist_cache[(e, s)] = d
        dist_cache[(s, s)] = 0.0

    population = []
    for _ in range(POP):
        chrom = [random.choice(truck_ids) for _ in range(num_tasks)]
        order = {tid: [] for tid in truck_ids}
        for i, tid in enumerate(chrom):
            order[tid].append(i)
        for tid in truck_ids:
            random.shuffle(order[tid])
        population.append((chrom, order))

    best_individual = None
    best_fitness = -1.0
    convergence_costs = []

    for it in range(ITERS):
        pc = PC_MAX - (PC_MAX - PC_MIN) * it / ITERS
        pm = PM_MAX - (PM_MAX - PM_MIN) * it / ITERS

        # 计算所有个体的原始指标和适应度
        raw_data = [_compute_fitness(c, o, trucks, tasks, dist_cache)
                    for c, o in population]
        all_dist = [r[0] for r in raw_data]
        all_ms = [r[1] for r in raw_data]
        max_dist = max(all_dist)
        max_ms = max(all_ms)

        # 标准化适应度
        fitnesses = []
        for (td, ms, _), (c, o) in zip(raw_data, population):
            norm_dist = td / max_dist if max_dist > 0 else 0
            norm_ms = ms / max_ms if max_ms > 0 else 0
            cost = 0.5 * norm_dist + 0.5 * norm_ms
            fitnesses.append(1.0 / (cost + 1e-4))

        iter_best = max(fitnesses)
        if iter_best > best_fitness:
            best_fitness = iter_best
            best_individual = population[fitnesses.index(iter_best)]
        # 保存标准化后的代价值（越小越好）
        convergence_costs.append(1.0 / (best_fitness + 1e-4))

        # 锦标赛选择
        new_pop = []
        # 精英保留：前 5 个
        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        for idx in sorted_idx[:5]:
            c = list(population[idx][0])
            o = {k: list(v) for k, v in population[idx][1].items()}
            new_pop.append((c, o))

        while len(new_pop) < POP:
            s1 = _tournament_select(population, fitnesses, k=3)
            s2 = _tournament_select(population, fitnesses, k=3)
            p1_c, p1_o = list(s1[0]), {k: list(v) for k, v in s1[1].items()}
            p2_c, p2_o = list(s2[0]), {k: list(v) for k, v in s2[1].items()}

            if random.random() < pc:
                pt = random.randint(1, num_tasks - 1)
                child_c = p1_c[:pt] + p2_c[pt:]
                child_o = {tid: [] for tid in truck_ids}
                for i, tid in enumerate(child_c):
                    child_o[tid].append(i)
                for tid in truck_ids:
                    random.shuffle(child_o[tid])
                _repair(child_c, child_o, num_tasks, truck_ids)
            else:
                child_c, child_o = list(p1_c), {k: list(v) for k, v in p1_o.items()}

            # per-gene truck-reassignment mutation
            mutated = False
            for idx in range(num_tasks):
                if random.random() < pm:
                    old_tid = child_c[idx]
                    new_tid = random.choice([t for t in truck_ids if t != old_tid])
                    child_c[idx] = new_tid
                    if idx in child_o[old_tid]:
                        child_o[old_tid].remove(idx)
                    child_o[new_tid].append(idx)
                    mutated = True
            if mutated:
                for tid in truck_ids:
                    random.shuffle(child_o[tid])
                _repair(child_c, child_o, num_tasks, truck_ids)

            # order-swap mutation
            if random.random() < pm * 2:
                tid = random.choice(truck_ids)
                seq = child_o[tid]
                if len(seq) >= 2:
                    i1, i2 = random.sample(range(len(seq)), 2)
                    seq[i1], seq[i2] = seq[i2], seq[i1]

            # 2-opt 局部搜索（低概率应用于每个子代）
            if random.random() < 0.1:
                _two_opt_improve(child_o, tasks, dist_cache)

            new_pop.append((child_c, child_o))

        population = new_pop

    best_c, best_o = best_individual
    assignment = {tid: [] for tid in truck_ids}
    for tid in truck_ids:
        for idx in best_o[tid]:
            assignment[tid].append(tasks[idx][0])

    return assignment, time.time() - start_time, convergence_costs


def run_ca_ga(graph, coords, trucks, tasks, path_fn):
    """拥堵感知遗传算法 (Congestion-Aware GA)。
    使用包含拥堵指数和空驶率的适应度函数。
    """
    import time
    start_time = time.time()
    truck_ids = [t[0] for t in trucks]
    num_tasks = len(tasks)
    POP = 200
    ITERS = 300
    PC_MAX, PC_MIN = 0.85, 0.5
    PM_MAX, PM_MIN = 0.2, 0.05

    # 预计算距离 + 路径
    node_ids = list(graph.keys())
    dist_cache = {}
    path_cache = {}
    for s in node_ids:
        for e in node_ids:
            if s != e and (s, e) not in dist_cache:
                p, d = path_fn(graph, coords, s, e)
                dist_cache[(s, e)] = d
                dist_cache[(e, s)] = d
                path_cache[(s, e)] = p
                path_cache[(e, s)] = list(reversed(p))
        dist_cache[(s, s)] = 0.0
        path_cache[(s, s)] = [s]

    population = []
    for _ in range(POP):
        chrom = [random.choice(truck_ids) for _ in range(num_tasks)]
        order = {tid: [] for tid in truck_ids}
        for i, tid in enumerate(chrom):
            order[tid].append(i)
        for tid in truck_ids:
            random.shuffle(order[tid])
        population.append((chrom, order))

    best_individual = None
    best_fitness = -1.0
    convergence_costs = []

    for it in range(ITERS):
        pc = PC_MAX - (PC_MAX - PC_MIN) * it / ITERS
        pm = PM_MAX - (PM_MAX - PM_MIN) * it / ITERS

        # 计算所有个体的原始指标
        raw_data = [_compute_ca_fitness(c, o, trucks, tasks, dist_cache, path_cache)
                    for c, o in population]
        all_dist = [r[0] for r in raw_data]
        all_ms = [r[1] for r in raw_data]
        all_cng = [r[2] for r in raw_data]
        max_dist = max(all_dist)
        max_ms = max(all_ms)
        max_cng = max(all_cng) if all_cng else 1

        # 标准化适应度
        fitnesses = []
        for (td, ms, cng, _), (c, o) in zip(raw_data, population):
            norm_dist = td / max_dist if max_dist > 0 else 0
            norm_ms = ms / max_ms if max_ms > 0 else 0
            norm_cng = cng / max_cng if max_cng > 0 else 0
            cost = 0.4 * norm_dist + 0.3 * norm_ms + 0.3 * norm_cng
            fitnesses.append(1.0 / (cost + 1e-4))

        iter_best = max(fitnesses)
        if iter_best > best_fitness:
            best_fitness = iter_best
            best_individual = population[fitnesses.index(iter_best)]
        convergence_costs.append(1.0 / (best_fitness + 1e-4))

        # 锦标赛选择 + 精英保留
        sorted_idx = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)
        new_pop = []
        for idx in sorted_idx[:5]:
            c = list(population[idx][0])
            o = {k: list(v) for k, v in population[idx][1].items()}
            new_pop.append((c, o))

        while len(new_pop) < POP:
            s1 = _tournament_select(population, fitnesses, k=3)
            s2 = _tournament_select(population, fitnesses, k=3)
            p1_c, p1_o = list(s1[0]), {k: list(v) for k, v in s1[1].items()}
            p2_c, p2_o = list(s2[0]), {k: list(v) for k, v in s2[1].items()}

            if random.random() < pc:
                pt = random.randint(1, num_tasks - 1)
                child_c = p1_c[:pt] + p2_c[pt:]
                child_o = {tid: [] for tid in truck_ids}
                for i, tid in enumerate(child_c):
                    child_o[tid].append(i)
                for tid in truck_ids:
                    random.shuffle(child_o[tid])
                _repair(child_c, child_o, num_tasks, truck_ids)
            else:
                child_c, child_o = list(p1_c), {k: list(v) for k, v in p1_o.items()}

            mutated = False
            for idx in range(num_tasks):
                if random.random() < pm:
                    old_tid = child_c[idx]
                    new_tid = random.choice([t for t in truck_ids if t != old_tid])
                    child_c[idx] = new_tid
                    if idx in child_o[old_tid]:
                        child_o[old_tid].remove(idx)
                    child_o[new_tid].append(idx)
                    mutated = True
            if mutated:
                for tid in truck_ids:
                    random.shuffle(child_o[tid])
                _repair(child_c, child_o, num_tasks, truck_ids)

            if random.random() < pm * 2:
                tid = random.choice(truck_ids)
                seq = child_o[tid]
                if len(seq) >= 2:
                    i1, i2 = random.sample(range(len(seq)), 2)
                    seq[i1], seq[i2] = seq[i2], seq[i1]

            new_pop.append((child_c, child_o))

        population = new_pop

    best_c, best_o = best_individual
    assignment = {tid: [] for tid in truck_ids}
    for tid in truck_ids:
        for idx in best_o[tid]:
            assignment[tid].append(tasks[idx][0])

    return assignment, time.time() - start_time, convergence_costs