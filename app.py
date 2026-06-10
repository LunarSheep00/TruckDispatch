import json, time, os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

TRUCK_COLORS = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
]
COORDS = {
    1:(0,0),2:(0,2),3:(0,4),4:(0,6),
    5:(2,0),6:(2,2),7:(2,4),8:(2,6),
    9:(4,0),10:(4,2),11:(4,4),12:(4,6),
    13:(6,0),14:(6,2),15:(6,4),16:(6,6),
    17:(8,0),18:(8,2),19:(8,4),20:(8,6),
    21:(10,0),22:(10,2),23:(10,4),24:(10,6),
    25:(12,0),26:(12,2),27:(12,4),28:(12,6),
    29:(14,0),30:(14,2),31:(14,4),32:(14,6),
    33:(16,0),34:(16,2),35:(16,4),36:(16,6),
    37:(18,0),38:(18,2),39:(18,4),40:(18,6),
}
EDGES = [
    # vertical edges within each column (2km)
    (1,2,2),(2,3,2),(3,4,2),
    (5,6,2),(6,7,2),(7,8,2),
    (9,10,2),(10,11,2),(11,12,2),
    (13,14,2),(14,15,2),(15,16,2),
    (17,18,2),(18,19,2),(19,20,2),
    (21,22,2),(22,23,2),(23,24,2),
    (25,26,2),(26,27,2),(27,28,2),
    (29,30,2),(30,31,2),(31,32,2),
    (33,34,2),(34,35,2),(35,36,2),
    (37,38,2),(38,39,2),(39,40,2),
    # horizontal edges: Y=0,2 行 3km, Y=4,6 行 5km
    (1,5,3),(2,6,3),(3,7,5),(4,8,5),
    (5,9,3),(6,10,3),(7,11,5),(8,12,5),
    (9,13,3),(10,14,3),(11,15,5),(12,16,5),
    (13,17,3),(14,18,3),(15,19,5),(16,20,5),
    (17,21,3),(18,22,3),(19,23,5),(20,24,5),
    (21,25,3),(22,26,3),(23,27,5),(24,28,5),
    (25,29,3),(26,30,3),(27,31,5),(28,32,5),
    (29,33,3),(30,34,3),(31,35,5),(32,36,5),
    (33,37,3),(34,38,3),(35,39,5),(36,40,5),
]

ALGO_NAMES = {
    "ga": "GA+A*",
    "greedy": "贪心+Dijkstra",
    "ca_ga": "CA-GA+A* (拥堵感知)",
    "nsga2": "NSGA-II+A* (多目标)",
    "q_learning": "Q-Learning+A*",
    "dqn": "DQN+A*",
}
ALGO_LIST = ["ga", "ca_ga", "nsga2", "q_learning", "dqn", "greedy"]
STATIC_ALGO_LIST = ["greedy", "ga", "ca_ga", "nsga2"]

DYNAMIC_ALGO_NAMES = {
    "greedy_dynamic": "贪心_Dynamic",
    "ga_dynamic": "GA_Dynamic",
    "q_learning": "Q-Learning",
    "dqn": "DQN",
}


def node_type(nid):
    if nid in {1,2,3,4,5,8,9,12}: return "berth"
    if nid in {38,39}: return "gate"
    return "warehouse"


def draw_frame(ax, truck_paths, algo_label, frame_idx, edge_usage=None, show_heat=False):
    """绘制动画帧，支持拥堵着色。"""
    ax.axvspan(-0.5, 4.5, alpha=0.15, color="steelblue")
    ax.axvspan(17.5, 18.5, alpha=0.15, color="green")

    # --- 计算当前帧的累计边使用量（用于拥堵着色动画） ---
    # 从 truck_paths 动态计算到 frame_idx 为止的累计使用量，
    # 使得每一帧的拥堵着色随动画推进而变化。
    frame_usage = {}
    if show_heat and truck_paths:
        for tid_str, path in truck_paths.items():
            seg = min(frame_idx, len(path) - 1)
            for j in range(seg):
                u, v = path[j], path[j + 1]
                key = (min(u, v), max(u, v))
                frame_usage[key] = frame_usage.get(key, 0) + 1

    max_usage = 1
    if frame_usage:
        vals = [v for v in frame_usage.values() if v > 0]
        max_usage = max(vals) if vals else 1

    for u, v, w in EDGES:
        x1, y1 = COORDS[u]
        x2, y2 = COORDS[v]
        if show_heat:
            usage = frame_usage.get((u, v), 0) + frame_usage.get((v, u), 0)
            ratio = usage / max_usage if max_usage > 0 else 0
            # 使用 RdYlGn_r 配色：红(堵)→黄→绿(畅通)
            cmap = plt.cm.RdYlGn_r
            color = cmap(ratio)
            lw = 1.5 + ratio * 3
        else:
            color = "k"
            lw = 1
        ax.plot([x1, x2], [y1, y2], "-", color=color, lw=lw, zorder=1)
        if not show_heat:
            ax.text((x1+x2)/2, (y1+y2)/2+0.1, f"{w}km", fontsize=6, ha="center", color="gray")

    # --- 画节点 ---
    for nid, (x, y) in COORDS.items():
        nt = node_type(nid)
        if nt == "berth":
            ax.plot(x, y, "o", color="steelblue", ms=12, zorder=3)
        elif nt == "warehouse":
            ax.plot(x, y, "s", color="darkorange", ms=10, zorder=3)
        else:
            ax.plot(x, y, "^", color="green", ms=10, zorder=3)
        ax.text(x, y+0.2, str(nid), fontsize=8, ha="center", fontweight="bold")

    # --- 画卡车 + 路径 ---
    # 热力模式：隐藏卡车路径线（避免遮挡 RdYlGn_r 拥堵着色）
    # 普通模式：显示彩色路径线
    truck_handles = []
    truck_paths_sorted = sorted(truck_paths.items(), key=lambda x: int(x[0]))
    for i, (tid_str, path) in enumerate(truck_paths_sorted):
        color = TRUCK_COLORS[i % len(TRUCK_COLORS)]
        if not path:
            continue
        seg = min(frame_idx, len(path) - 1)
        x, y = COORDS[path[seg]]
        ax.plot(x, y, "o", color=color, ms=14, zorder=5)
        ax.text(x, y, "T" + str(tid_str), fontsize=7, ha="center",
                va="center", color="white", fontweight="bold", zorder=6)
        if not show_heat:
            xs = [COORDS[path[j]][0] for j in range(seg + 1)]
            ys = [COORDS[path[j]][1] for j in range(seg + 1)]
            ax.plot(xs, ys, "-", color=color, lw=2, alpha=0.6, zorder=2)
        truck_handles.append(mpatches.Patch(color=color, label=f"车辆{tid_str}"))

    # --- 拥堵热力图 colorbar ---
    if show_heat and edge_usage:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        norm = mcolors.Normalize(vmin=0, vmax=1)
        sm = cm.ScalarMappable(norm=norm, cmap=cm.RdYlGn_r)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.set_label('拥堵程度 (红=高, 绿=低)', fontsize=8)

    # --- 图例 ---
    berth_p = mpatches.Patch(color="steelblue", label="泊位")
    wh_p = mpatches.Patch(color="darkorange", label="仓库")
    gate_p = mpatches.Patch(color="green", label="闸口")
    ax.legend(handles=[berth_p, wh_p, gate_p] + truck_handles,
              loc="upper right", fontsize=7)
    ax.set_xlim(-1, 19)
    ax.set_ylim(-1, 7)
    ax.set_title(f"{algo_label} — 帧 {frame_idx+1}" +
                 (" [拥堵热力模式]" if show_heat else ""), fontsize=10)


def main():
    st.set_page_config(page_title="港口集卡智能调度系统", layout="wide")
    st.title("港口集卡智能调度系统")
    st.markdown("*基于图搜索、智能优化与强化学习的港口集卡协同调度研究*")

    has_static = os.path.exists("images/results.json")
    has_dynamic = os.path.exists("images/results_dynamic.json")

    if not has_static and not has_dynamic:
        st.error("results.json 和 results_dynamic.json 均不存在，请先运行 main.ipynb")
        return

    # --- 场景选择 ---
    with st.sidebar:
        st.header("场景选择")
        scene = st.radio("选择场景", ["静态场景", "动态场景"],
                         index=0, horizontal=True)

        if scene == "动态场景" and has_dynamic:
            with open("images/results_dynamic.json", encoding="utf-8") as f:
                data = json.load(f)
            algo_list_dynamic = [k for k in data.keys() if k in DYNAMIC_ALGO_NAMES]
            algo_list = algo_list_dynamic
            algo_name_func = lambda k: DYNAMIC_ALGO_NAMES.get(k, k)
        else:
            if not has_static:
                st.warning("静态结果不存在，加载动态结果")
                with open("images/results_dynamic.json", encoding="utf-8") as f:
                    data = json.load(f)
                algo_list = list(data.keys())
            else:
                with open("images/results.json", encoding="utf-8") as f:
                    data = json.load(f)
                algo_list = STATIC_ALGO_LIST
            algo_name_func = lambda k: ALGO_NAMES.get(k, k)

        st.header("算法选择")
        algo_key = st.selectbox(
            "选择算法查看动画和指标",
            algo_list,
            format_func=algo_name_func,
        )
        result = data.get(algo_key, {})
        algo_label = algo_name_func(algo_key)

        st.subheader("核心指标")
        col1, col2 = st.columns(2)
        col1.metric("总距离(km)", f"{result.get('total_dist', 0):.2f}")
        col2.metric("完工时间(h)", f"{result.get('makespan', 0):.4f}")
        col1.metric("运行时间(s)", f"{result.get('runtime', 0):.4f}")
        col2.metric("空驶率", f"{result.get('empty_ratio', 0):.3f}")
        if result.get('congestion'):
            st.metric("拥堵指数", f"{result['congestion']:.2f}")

        st.subheader("任务分配")
        assignment = result.get("assignment", {})
        for tid_str in sorted(assignment.keys(), key=lambda x: int(x)):
            task_ids = assignment[tid_str]
            with st.expander(f"车辆 {tid_str}"):
                st.write(f"任务: {task_ids}" if task_ids else "无任务")

        st.subheader("显示选项")
        show_heat = st.checkbox("显示拥堵热力", value=True,
                                help="边按拥堵程度从绿→黄→红着色")
        speed = st.slider("动画速度", 0.5, 3.0, 1.0, 0.25)

    # --- 左侧：动画 ---
    left, right = st.columns([3, 2])

    with left:
        st.subheader(f"{algo_label} 动画")
        truck_paths = result.get("truck_paths", {})
        edge_usage = result.get("edge_usage", None)
        max_frames = max((len(p) for p in truck_paths.values()), default=1)

        if "playing" not in st.session_state:
            st.session_state.playing = False
        if "frame" not in st.session_state:
            st.session_state.frame = 0
        if "algo" not in st.session_state:
            st.session_state.algo = algo_key
        elif st.session_state.algo != algo_key:
            st.session_state.frame = 0
            st.session_state.playing = False
            st.session_state.algo = algo_key

        bc1, bc2, bc3 = st.columns(3)
        if bc1.button("▶ 播放" if not st.session_state.playing else "⏸ 暂停"):
            st.session_state.playing = not st.session_state.playing
        if bc2.button("⏹ 重置"):
            st.session_state.frame = 0
            st.session_state.playing = False
        bc3.progress(st.session_state.frame / max(max_frames - 1, 1),
                     text=f"{st.session_state.frame+1}/{max_frames}")

        placeholder = st.empty()
        fig, ax = plt.subplots(figsize=(7, 4))
        draw_frame(ax, truck_paths, algo_label,
                   st.session_state.frame, edge_usage, show_heat)
        placeholder.pyplot(fig)
        plt.close()

        if st.session_state.playing:
            if st.session_state.frame < max_frames - 1:
                st.session_state.frame += 1
                time.sleep(0.5 / speed)
                st.rerun()
            else:
                st.session_state.playing = False

    # --- 右侧：静态图表 ---
    with right:
        # 拥堵热力图
        st.subheader("拥堵热力图 (GA vs CA-GA)")
        try:
            heatmap_dir = "images"
            heatmap_files = [f for f in os.listdir(heatmap_dir) if f.startswith("拥堵热力图") and f.endswith(".png")]
            if heatmap_files:
                latest = max(heatmap_files, key=lambda f: os.path.getctime(os.path.join(heatmap_dir, f)))
                st.image(os.path.join(heatmap_dir, latest), caption="拥堵热力图参考 (红=高拥堵, 绿=低拥堵)", width=500)
        except Exception:
            pass

        st.subheader("算法多指标对比")

        ALGO_COLORS = ["#aaaaaa", "#4363d8", "#3cb44b", "#f58231", "#e6194b"]

        def _algo_bar_chart(key, label, fmt=".2f"):
            """画单个指标的五算法柱状图。"""
            names, values = [], []
            for k in ALGO_LIST:
                r = data.get(k)
                if r and key in r:
                    names.append(ALGO_NAMES.get(k, k))
                    values.append(r[key])
            if not values:
                return
            fig, ax = plt.subplots(figsize=(5, 2))
            colors = ALGO_COLORS[:len(names)]
            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8)
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(names, fontsize=6, rotation=15)
            ax.set_ylabel(label, fontsize=8)
            ax.set_title(label, fontsize=9)
            for bar, v in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()*1.02,
                        f"{v:{fmt}}", ha="center", fontsize=6)
            st.pyplot(fig)
            plt.close()

        _algo_bar_chart("total_dist", "总距离 (km)")
        _algo_bar_chart("makespan", "完工时间 (h)")
        _algo_bar_chart("empty_ratio", "空驶率", ".3f")

        # GA 收敛曲线
        ga_conv = data.get("ga", {}).get("convergence", [])
        if ga_conv:
            st.subheader("GA 收敛曲线")
            fig3, ax3 = plt.subplots(figsize=(5, 2.5))
            step = max(1, len(ga_conv) // 60)
            sampled = ga_conv[::step]
            x_sampled = range(0, len(ga_conv), step)[:len(sampled)]
            ax3.plot(x_sampled, sampled, color="#e6194b", lw=1, alpha=0.3, label="代价值")
            window_smooth = max(1, len(sampled) // 10)
            if len(sampled) >= window_smooth:
                import numpy as np
                smooth = np.convolve(sampled, np.ones(window_smooth)/window_smooth, mode='valid')
                ax3.plot(x_sampled[window_smooth-1:], smooth, color="#e6194b", lw=2.5, label="趋势线")
            ax3.set_xlabel("迭代次数", fontsize=8)
            ax3.set_ylabel("代价值 (越小越好)", fontsize=8)
            ax3.legend(fontsize=7)
            ax3.grid(True, alpha=0.3)
            st.pyplot(fig3)
            plt.close()

        # RL 训练曲线（当前算法的 reward_history）
        reward_data = result.get("reward_history", [])
        if reward_data:
            algo_label_short = ALGO_NAMES.get(algo_key, algo_key)
            st.subheader(f"{algo_label_short} 训练曲线")
            fig4, ax4 = plt.subplots(figsize=(5, 2.5))
            ax4.plot(range(len(reward_data)), reward_data,
                     alpha=0.3, color="steelblue", lw=0.6, label="原始")
            window = max(1, len(reward_data)//20)
            if len(reward_data) >= window:
                import numpy as np
                smooth = np.convolve(reward_data, np.ones(window)/window, mode='valid')
                ax4.plot(range(window, len(reward_data)+1), smooth,
                         color="red", lw=2, label="平滑")
            ax4.set_xlabel("Episode", fontsize=8)
            ax4.set_ylabel("总奖励", fontsize=8)
            ax4.legend(fontsize=7, loc='upper left', framealpha=0.7)
            ax4.grid(True, alpha=0.3)
            st.pyplot(fig4)
            plt.close()


if __name__ == "__main__":
    main()
