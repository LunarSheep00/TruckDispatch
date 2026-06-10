import json, os
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

_OUTPUT_DIR = "images"
os.makedirs(_OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

_BERTHS = {1,2,3,4,5,8,9,12}
_GATES  = {38,39}

def _node_type(nid):
    if nid in _BERTHS: return "berth"
    if nid in _GATES: return "gate"
    return "warehouse"

# ---------- 基础指标 ----------

def compute_metrics(assignment, truck_paths, trucks, tasks, edge_usage=None, dist_cache=None):
    """计算指标，支持真实空驶率和拥堵指数。

    Args:
        assignment: {truck_id: [task_id, ...]}
        truck_paths: {truck_id: [node, ...]}
        trucks: [(id, start, speed, status), ...]
        tasks: [(id, start, end), ...] 或 [(id, start, end, arrival), ...]
        edge_usage: 边使用量统计（可选，用于拥堵指数）
        dist_cache: 距离缓存（可选，用于真实空驶率）
    """
    speed = trucks[0][2]
    task_dict = {t[0]: (t[1], t[2]) for t in tasks}
    truck_start = {t[0]: t[1] for t in trucks}
    truck_dists = {}
    for tid, path in truck_paths.items():
        if dist_cache:
            # 用 dist_cache 逐边求和（支持非均匀边权）
            d = 0.0
            for i in range(len(path) - 1):
                d += dist_cache.get((path[i], path[i+1]), 2.0)
            truck_dists[tid] = d
        else:
            # 回退方案（假设全 2km）
            truck_dists[tid] = (len(path) - 1) * 2.0
    total_dist = sum(truck_dists.values())
    avg_dist = total_dist / len(trucks)
    makespan = max(d / speed for d in truck_dists.values()) if truck_dists else 0.0

    # 空驶率（如果提供了 dist_cache 则真实计算）
    empty_ratio = 0.0
    if dist_cache:
        total_empty = 0.0
        for tid, task_ids in assignment.items():
            pos = truck_start[tid]
            for jid in task_ids:
                js, je = task_dict.get(jid, (pos, pos))
                total_empty += dist_cache.get((pos, js), 0)
                pos = je
        empty_ratio = total_empty / total_dist if total_dist > 0 else 0.0

    # 拥堵指数
    congestion = 0.0
    if edge_usage:
        vals = list(edge_usage.values())
        if vals:
            max_u = max(vals)
            thresh = max_u * 0.7
            congestion = sum((u - thresh) ** 2 for u in vals if u > thresh)

    return {
        "total_dist": total_dist,
        "avg_dist": avg_dist,
        "makespan": makespan,
        "empty_ratio": empty_ratio,
        "congestion": congestion,
        "truck_paths": truck_paths,
        "assignment": assignment,
    }


# ---------- 热力图 ----------

def _draw_network(ax, coords, graph, edge_usages=None):
    """画出港口路网，可选边着色。"""
    max_usage = 1
    if edge_usages:
        vals = [v for v in edge_usages.values() if v > 0]
        max_usage = max(vals) if vals else 1

    for u, neighbors in graph.items():
        for v, w in neighbors.items():
            if u < v:
                x1, y1 = coords[u]
                x2, y2 = coords[v]
                if edge_usages:
                    usage = edge_usages.get((u, v), 0) + edge_usages.get((v, u), 0)
                    ratio = usage / max_usage if max_usage > 0 else 0
                    # 使用 RdYlGn_r 配色（红=堵，绿=畅），与 app.py 一致
                    color = plt.cm.RdYlGn_r(ratio)
                    lw = 1.5 + ratio * 3
                else:
                    color = "k"
                    lw = 1
                ax.plot([x1, x2], [y1, y2], "-", color=color, lw=lw, zorder=1)
                if not edge_usages:
                    ax.text((x1+x2)/2, (y1+y2)/2+0.1, f"{w:.0f}km",
                            fontsize=6, ha="center", color="gray")

    # 画节点
    for nid, (x, y) in coords.items():
        nt = _node_type(nid)
        if nt == "berth":
            ax.plot(x, y, "o", color="steelblue", ms=12, zorder=3)
        elif nt == "warehouse":
            ax.plot(x, y, "s", color="darkorange", ms=10, zorder=3)
        else:
            ax.plot(x, y, "^", color="green", ms=10, zorder=3)
        ax.text(x, y+0.2, str(nid), fontsize=8, ha="center", fontweight="bold")


def save_heatmap(ga_edge_usage, ca_edge_usage, graph, coords, date_str):
    """保存拥堵热力图对比（GA vs CA-GA）。"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, usage, title in zip(axes,
                                [ga_edge_usage, ca_edge_usage],
                                ["GA+A* 边使用量", "CA-GA+A* 边使用量"]):
        _draw_network(ax, coords, graph, usage)
        ax.set_title(title, fontsize=13)
        ax.set_xlim(-1, 19)
        ax.set_ylim(-1, 7)

    plt.suptitle("港口路网拥堵热力图对比", fontsize=15)

    # 节点图例（全局，放在底部）
    import matplotlib.patches as mpatches
    berth_p = mpatches.Patch(color="steelblue", label="泊位")
    wh_p = mpatches.Patch(color="darkorange", label="仓库")
    gate_p = mpatches.Patch(color="green", label="闸口")
    fig.legend(handles=[berth_p, wh_p, gate_p], loc='lower center',
               ncol=3, fontsize=11, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))

    # 添加 colorbar
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
    norm = mcolors.Normalize(vmin=0, vmax=1)
    sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
    sm.set_array([])
    fig.subplots_adjust(right=0.88, bottom=0.12)
    cbar_ax = fig.add_axes([0.89, 0.15, 0.012, 0.7])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label('拥堵程度 (红=高, 绿=低)', fontsize=11)

    plt.savefig(os.path.join(_OUTPUT_DIR, f"拥堵热力图_{date_str}.png"), dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  ✓ 拥堵热力图 → {_OUTPUT_DIR}/拥堵热力图_{date_str}.png")


# ---------- Pareto 前沿 ----------

def save_pareto_front(pareto_front, date_str):
    """保存 NSGA-II Pareto 前沿图。"""
    if not pareto_front:
        return

    dists = [p["total_dist"] for p in pareto_front]
    makespans = [p["makespan"] for p in pareto_front]
    empties = [p["empty_ratio"] for p in pareto_front]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：距离 vs 完工时间
    ax = axes[0]
    ax.scatter(dists, makespans, c="steelblue", s=30, alpha=0.7, label="Pareto 最优解")
    pareto_dists = sorted(dists)
    pareto_ms = [makespans[dists.index(d)] for d in pareto_dists]
    ax.plot(pareto_dists, pareto_ms, "r--", alpha=0.5, lw=1.5, label="Pareto 前沿")
    ax.set_xlabel("总距离 (km)")
    ax.set_ylabel("完工时间 (h)")
    ax.set_title("Pareto 前沿: 总距离 vs 完工时间")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 右图：距离 vs 空驶率
    ax = axes[1]
    ax.scatter(dists, empties, c="darkorange", s=30, alpha=0.7, label="Pareto 最优解")
    ax.set_xlabel("总距离 (km)")
    ax.set_ylabel("空驶率")
    ax.set_title("Pareto 前沿: 总距离 vs 空驶率")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.suptitle("NSGA-II 多目标优化 Pareto 前沿", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, f"帕累托前沿_{date_str}.png"), dpi=100)
    plt.close()
    print(f"  ✓ Pareto 前沿 → {_OUTPUT_DIR}/帕累托前沿_{date_str}.png")


# ---------- RL 学习曲线 ----------

def save_rl_curve(reward_history, date_str):
    """保存 Q-Learning 训练曲线。"""
    if not reward_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    episodes = range(1, len(reward_history) + 1)
    ax.plot(episodes, reward_history, alpha=0.4, color="steelblue", lw=0.8, label="每轮奖励")

    # 滑动平均
    window = max(1, len(reward_history) // 20)
    if len(reward_history) >= window:
        smoothed = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        ax.plot(range(window, len(reward_history) + 1), smoothed,
                color="red", lw=2, label=f"滑动平均 (窗口={window})")

    ax.set_xlabel("Episode")
    ax.set_ylabel("总奖励")
    ax.set_title("Q-Learning 训练曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, f"学习曲线_{date_str}.png"), dpi=100)
    plt.close()
    print(f"  ✓ RL 学习曲线 → {_OUTPUT_DIR}/学习曲线_{date_str}.png")


# ---------- 五算法对比 ----------

def save_comparison_chart(results, date_str):
    """保存五算法对比柱状图。"""
    algo_names = [r["name"] for r in results]
    colors = ["#aaaaaa", "#4363d8", "#3cb44b", "#f58231", "#e6194b"]

    metrics_keys = [
        ("total_dist", "总距离 (km)"),
        ("makespan", "完工时间 (h)"),
        ("runtime", "运行时间 (s)"),
        ("empty_ratio", "空驶率"),
        ("congestion", "拥堵指数"),
    ]

    n_metrics = len(metrics_keys)
    fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 3.5, 4.5))

    for ax, (key, label) in zip(axes, metrics_keys):
        values = [r.get(key, 0) for r in results]
        bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(algo_names, fontsize=7, rotation=15)
        ax.set_title(label, fontsize=10)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.02,
                    f"{v:.2f}" if v < 1000 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)

    plt.suptitle("五种算法指标对比", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, f"算法对比图_{date_str}.png"), dpi=100)
    plt.close()
    print(f"  ✓ 五算法对比 → {_OUTPUT_DIR}/算法对比图_{date_str}.png")


# ---------- 保存结果 ----------

def save_results(results_dict, path=None):
    """保存结果为 JSON（兼容旧版 + 新版格式）。"""
    if path is None:
        path = os.path.join(_OUTPUT_DIR, "results.json")
    def serialize(name, r):
        return {
            "name": name,
            "total_dist": r.get("total_dist", 0),
            "avg_dist": r.get("avg_dist", 0),
            "makespan": r.get("makespan", 0),
            "empty_ratio": r.get("empty_ratio", 0),
            "congestion": r.get("congestion", 0),
            "runtime": r.get("runtime", 0),
            "convergence": r.get("convergence", []),
            "reward_history": r.get("reward_history", []),
            "pareto_front": r.get("pareto_front", []),
            "assignment": {str(k): v for k, v in r.get("assignment", {}).items()},
            "truck_paths": {str(k): v for k, v in r.get("truck_paths", {}).items()},
            "edge_usage": {str(k): v for k, v in r.get("edge_usage", {}).items()},
        }
    serialized = {}
    for name, result in results_dict.items():
        serialized[name] = serialize(name, result)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 结果已保存 → {path}")


# ---------- 原有图表（兼容扩展） ----------

def save_charts(results_dict, graph, coords):
    """增强版图表保存（包含所有新图表）。"""
    date_str = datetime.datetime.now().strftime("%Y%m%d")

    ga = results_dict.get("ga", {})
    greedy = results_dict.get("greedy", {})
    ca_ga = results_dict.get("ca_ga", {})
    nsga2 = results_dict.get("nsga2", {})
    ql = results_dict.get("q_learning", {})

    # 原有 GA 路径图
    if ga.get("truck_paths"):
        _save_ga_paths(ga, graph, coords, date_str)

    # 原有收敛曲线
    if ga.get("convergence"):
        _save_convergence(ga, date_str)

    # 1. 拥堵热力图
    ga_usage = ga.get("edge_usage", {})
    ca_usage = ca_ga.get("edge_usage", {})
    if ga_usage or ca_usage:
        try:
            save_heatmap(ga_usage, ca_usage, graph, coords, date_str)
        except Exception as e:
            print(f"  ⚠ 热力图生成失败: {e}")

    # 2. Pareto 前沿
    pf = nsga2.get("pareto_front", [])
    if pf:
        try:
            save_pareto_front(pf, date_str)
        except Exception as e:
            print(f"  ⚠ Pareto前沿生成失败: {e}")

    # 3. RL 学习曲线
    rh = ql.get("reward_history", [])
    if rh:
        try:
            save_rl_curve(rh, date_str)
        except Exception as e:
            print(f"  ⚠ 学习曲线生成失败: {e}")

    # 4. 五算法对比
    all_results = []
    for name, r in [("贪心+Dijkstra", greedy), ("GA+A*", ga),
                     ("CA-GA+A*", ca_ga), ("NSGA-II+A*", nsga2),
                     ("Q-Learning", ql)]:
        if r:
            all_results.append({"name": name, **r})
    if len(all_results) >= 2:
        try:
            save_comparison_chart(all_results, date_str)
        except Exception as e:
            print(f"  ⚠ 对比图生成失败: {e}")


# ---------- 原有图表子函数（保持向后兼容） ----------

def _save_ga_paths(ga_result, graph, coords, date_str):
    """保存 GA 路径图（原有逻辑）。"""
    truck_colors = ["#e6194b","#3cb44b","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axvspan(-0.5, 0.5, alpha=0.15, color="steelblue", label="水域")

    for u, neighbors in graph.items():
        for v, w in neighbors.items():
            if u < v:
                x1,y1 = coords[u]; x2,y2 = coords[v]
                ax.plot([x1,x2],[y1,y2],"k-",lw=1,zorder=1)
                ax.text((x1+x2)/2,(y1+y2)/2+0.1,f"{w:.0f}km",fontsize=6,ha="center",color="gray")

    for nid,(x,y) in coords.items():
        nt = _node_type(nid)
        if nt=="berth": ax.plot(x,y,"o",color="steelblue",ms=12,zorder=3)
        elif nt=="warehouse": ax.plot(x,y,"s",color="darkorange",ms=10,zorder=3)
        else: ax.plot(x,y,"^",color="green",ms=10,zorder=3)
        ax.text(x,y+0.2,str(nid),fontsize=8,ha="center",fontweight="bold")

    for i,(tid,path) in enumerate(ga_result["truck_paths"].items()):
        color = truck_colors[i % len(truck_colors)]
        xs=[coords[n][0] for n in path]; ys=[coords[n][1] for n in path]
        ax.plot(xs,ys,"-",color=color,lw=1.5,alpha=0.7,label=f"车辆{tid}",zorder=2)

    ax.set_title("港口路网图 (GA+A* 路径)", fontsize=13)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlim(-1, 19); ax.set_ylim(-1, 7)
    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, f"港口路网图_{date_str}.png"), dpi=100)
    plt.close()


def _save_metrics_comparison(ga_result, greedy_result, date_str):
    """保存指标对比柱状图（原有逻辑）。"""
    metrics = ["总距离(km)","平均距离(km)","完工时间(h)","运行时间(s)"]
    ga_vals = [ga_result.get("total_dist",0), ga_result.get("avg_dist",0),
               ga_result.get("makespan",0), ga_result.get("runtime",0)]
    gr_vals = [greedy_result.get("total_dist",0), greedy_result.get("avg_dist",0),
               greedy_result.get("makespan",0), greedy_result.get("runtime",0)]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(metrics)); w = 0.35
    bars1 = ax.bar([i-w/2 for i in x], ga_vals, w, label="GA+A*", color="#4363d8")
    bars2 = ax.bar([i+w/2 for i in x], gr_vals, w, label="贪心+Dijkstra", color="#f58231")
    ax.set_xticks(list(x)); ax.set_xticklabels(metrics)
    ax.set_title("算法指标对比", fontsize=13)
    ax.legend()
    for bar in list(bars1)+list(bars2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.1,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, f"指标对比图_{date_str}.png"), dpi=100)
    plt.close()


def _save_convergence(ga_result, date_str):
    """保存 GA 收敛曲线（稀疏采样 + 滑动平均平滑）。"""
    import numpy as np
    fig, ax = plt.subplots(figsize=(10, 6))
    conv = ga_result.get("convergence", [])
    step = max(1, len(conv) // 60)
    sampled = conv[::step]
    x_sampled = range(0, len(conv), step)[:len(sampled)]
    # 原始采样（半透明）
    ax.plot(x_sampled, sampled, color="#e6194b", lw=1, alpha=0.3, label="代价值")
    # 滑动平均趋势线
    window_smooth = max(1, len(sampled) // 10)
    if len(sampled) >= window_smooth:
        smooth = np.convolve(sampled, np.ones(window_smooth)/window_smooth, mode='valid')
        ax.plot(x_sampled[window_smooth-1:], smooth, color="#e6194b", lw=2.5, label="趋势线")
    ax.set_xlabel("迭代次数"); ax.set_ylabel("代价值 (越小越好)")
    ax.set_title("GA 收敛曲线", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(_OUTPUT_DIR, f"收敛曲线图_{date_str}.png"), dpi=100)
    plt.close()


def print_summary(results_dict):
    """打印所有算法的结果汇总。"""
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    lines = ["="*60, "港口集卡智能调度系统 — 运行结果", "="*60]

    for name, r in results_dict.items():
        if not r:
            continue
        lines.append(f"\n【{name}】")
        lines.append(f"  总距离: {r.get('total_dist', 0):.2f} km")
        lines.append(f"  平均距离: {r.get('avg_dist', 0):.2f} km")
        lines.append(f"  完工时间: {r.get('makespan', 0):.4f} h")
        lines.append(f"  运行时间: {r.get('runtime', 0):.4f} s")
        if 'empty_ratio' in r:
            lines.append(f"  空驶率: {r['empty_ratio']:.3f}")
        if 'congestion' in r and r['congestion']:
            lines.append(f"  拥堵指数: {r['congestion']:.2f}")

    lines += ["\n"+"="*60, "任务分配方案", "="*60]
    for name, r in results_dict.items():
        if not r or not r.get("assignment"):
            continue
        lines.append(f"\n【{name}】")
        for tid_str, task_ids in r["assignment"].items():
            path = r.get("truck_paths", {}).get(tid_str, [])
            lines.append(f"  车辆{tid_str}: 任务{task_ids}")

    text = "\n".join(lines)
    print(text)
    with open(os.path.join(_OUTPUT_DIR, f"路径方案_{date_str}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    with open(os.path.join(_OUTPUT_DIR, f"指标对比_{date_str}.txt"), "w", encoding="utf-8") as f:
        f.write(text)
    print("\n所有输出已保存。")
