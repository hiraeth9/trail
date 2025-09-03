
# -*- coding: utf-8 -*-
import os, sys, argparse, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
ALGOS={'sheer': ('algs.sheer','SHEER'),'actar': ('algs.actar','ACTAR'),
       'tfsm_dpc': ('algs.tfsm_dpc','TFSM_DPC'),'dst_woa': ('algs.dst_woa','DST_WOA'),
       'trail': ('algs.trail','TRAIL')}
# === 指标配置（名称、是否百分比、单位、派生/换算） ===
METRIC_SPECS = {
    # 可靠性
    "pdr":                      {"label": "PDR",                           "percent": True,  "unit": "%"},
    "drop_rate":                {"label": "Drop Rate",                     "percent": True,  "unit": "%"},
    "timely_transfer_rate":     {"label": "Timely Transfer Rate",          "percent": True,  "unit": "%"},
    # 寿命
    "FND":                      {"label": "FND (Rounds)",                  "percent": False, "unit": "rounds"},
    "HND":                      {"label": "HND (Rounds)",                  "percent": False, "unit": "rounds"},
    "LND":                      {"label": "LND (Rounds)",                  "percent": False, "unit": "rounds"},
    "func_life_pdr85":          {"label": "Functional Lifetime (PDR≥0.85)","percent": False, "unit": "rounds"},
    # 能耗/吞吐（派生列）
    "energy_mJ_per_pkt":        {"label": "Energy per Delivered",          "percent": False, "unit": "mJ/packet"},
    "throughput_kbit_round":    {"label": "Throughput",                    "percent": False, "unit": "kbit/round"},
    "energy_rate":              {"label": "Energy Rate",                   "percent": True,  "unit": "%"},  # 若本列是比例
    # 拓扑/结构
    "avg_hops_to_bs":           {"label": "Average Hops to BS",            "percent": False, "unit": ""},
    "avg_ch_per_round":         {"label": "Avg CH per Round",              "percent": False, "unit": ""},
    "avg_cluster_size":         {"label": "Average Cluster Size",          "percent": False, "unit": ""},
    # 安全/黑名单
    "malicious_drop":           {"label": "Malicious Drops",               "percent": False, "unit": "pkts"},
    "malicious_delay":           {"label": "Malicious Delay",               "percent": False, "unit": "pkts"},
    "blacklisted_malicious":    {"label": "Blacklisted Malicious",         "percent": False, "unit": "nodes"},
    "blacklisted_normal":       {"label": "Blacklisted Normal (FP)",       "percent": False, "unit": "nodes"},
    "false_blacklist_events":   {"label": "False Blacklist Events",        "percent": False, "unit": "events"},
    # 控制开销（派生列）
    "control_overhead_kbit":    {"label": "Control Overhead",              "percent": False, "unit": "kbit"},
}

def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """派生/换算一些更易读的列，不改变原 df"""
    d = df.copy()
    if "energy_per_delivered" in d.columns and "energy_mJ_per_pkt" not in d.columns:
        d["energy_mJ_per_pkt"] = d["energy_per_delivered"] * 1e3  # J -> mJ
    if "throughput_bits_per_round" in d.columns and "throughput_kbit_round" not in d.columns:
        d["throughput_kbit_round"] = d["throughput_bits_per_round"] / 1000.0
    if "control_overhead_bits" in d.columns and "control_overhead_kbit" not in d.columns:
        d["control_overhead_kbit"] = d["control_overhead_bits"] / 1000.0
    return d

def get_algo_ctor(tag:str):
    mod_name, cls = ALGOS[tag]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls)

def run_one(tag:str, n_nodes:int, n_malicious:int, seed:int, rounds:int, until_dead:bool, out_dir:str, suffix:str=""):
    from core.wsn_core import Simulation
    ctor = get_algo_ctor(tag)
    sim = Simulation(ctor, n_nodes=n_nodes, n_malicious=n_malicious, seed=seed)
    if until_dead:
        res, hist = sim.run(rounds=0)  # 0 => run until LND
    else:
        res, hist = sim.run(rounds=rounds)
    if suffix:
        hist.to_csv(os.path.join(out_dir, f'hist_{suffix}_{tag}.csv'), index=False)
    else:
        hist.to_csv(os.path.join(out_dir, f'hist_{tag}.csv'), index=False)
    res['seed']=seed
    return res

def run_all_tags(tags, n_nodes:int, n_malicious:int, seeds:list, rounds:int, until_dead:bool, out_dir:str, suffix:str=""):
    os.makedirs(out_dir, exist_ok=True)
    rows=[]
    for seed in seeds:
        for tag in tags:
            print(f'Running {tag} (seed={seed}) ...')
            res = run_one(tag, n_nodes, n_malicious, seed, rounds, until_dead, out_dir, suffix=suffix)
            rows.append(res)
    df = pd.DataFrame(rows)
    df.insert(0, 'algo_tag', [r['algo'] for r in rows])
    if suffix:
        df.to_csv(os.path.join(out_dir, f'summary_{suffix}.csv'), index=False)
    else:
        df.to_csv(os.path.join(out_dir, f'summary.csv'), index=False)
    return df

# >>> CHANGED: 更稳的聚合方式（均值±标准差），顺序固定，误差线与数值标注
# 覆盖原 plot_bars：按算法聚合画“柱状对比”，自动单位/百分比/CI
def plot_bars(summary: pd.DataFrame, out_dir: str, algo_order=None):
    summary = add_derived_columns(summary)
    if algo_order is None:
        algo_order = ["SHEER-2025", "ACTAR-2024", "TFSM-DPC-2024", "DST-WOA-2024", "TRAIL (ours)"]

    # 仅保留出现过的算法顺序
    algos_present = [a for a in algo_order if a in summary["algo"].unique()]
    if not algos_present:
        print("[warn] no known algos present in summary.")
        return

    grouped = summary.groupby('algo')
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(numeric_only=True).fillna(0.0)
    cnts  = grouped.size()

    # 想画的指标（从 METRIC_SPECS 里挑存在于数据里的键）
    keys = [k for k in METRIC_SPECS.keys() if k in means.columns]

    for key in keys:
        spec = METRIC_SPECS[key]
        y = means.loc[algos_present, key].values
        sd = stds.loc[algos_present, key].values if cnts.max() > 1 else None
        n  = cnts.loc[algos_present].values if cnts.max() > 1 else None

        # 百分比转 %
        y_plot = y * 100.0 if spec["percent"] else y
        yerr = None
        if sd is not None and n is not None:
            sem = sd / np.sqrt(np.maximum(n, 1))
            ci95 = 1.96 * sem
            yerr = ci95 * (100.0 if spec["percent"] else 1.0)

        plt.figure(figsize=(8, 4.2))
        xs = np.arange(len(algos_present))
        bars = plt.bar(xs, y_plot, yerr=yerr, capsize=4)
        plt.xticks(xs, algos_present, rotation=20, ha='right')
        ylabel = f'{spec["label"]} ({spec["unit"]})' if spec["unit"] else spec["label"]
        plt.ylabel(ylabel)
        plt.title(spec["label"])
        if spec["percent"]:
            plt.ylim(0, 100)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # 数值标注
        for i, b in enumerate(bars):
            val = y_plot[i]
            s = f'{val:.1f}' if spec["percent"] else (f'{val:.3g}' if abs(val) < 1000 else f'{val:,.0f}')
            plt.text(b.get_x()+b.get_width()/2, b.get_height()*1.01, s, ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'metric_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'metric_{key}.svg'), dpi=150)
        plt.close()


# 覆盖原 plot_lines_with_error：随比例（%）的“折线+95%CI 阴影”，自动单位/百分比
def plot_lines_with_error(df_all: pd.DataFrame, out_dir: str, seeds: int):
    df_all = add_derived_columns(df_all)
    algo_order = ["SHEER-2025", "ACTAR-2024", "TFSM-DPC-2024", "DST-WOA-2024", "TRAIL (ours)"]

    # 只画在 df 中实际存在的指标
    keys = [k for k in METRIC_SPECS.keys() if k in df_all.columns]

    def agg_mean_ci(df: pd.DataFrame, key: str):
        g = df.groupby(['algo', 'ratio'])[key]
        stat = g.agg(['mean', 'std', 'count']).reset_index()
        stat.rename(columns={'count':'n'}, inplace=True)
        stat['sem'] = stat['std'] / np.sqrt(np.maximum(stat['n'], 1))
        stat['ci95'] = 1.96 * stat['sem']
        return stat.sort_values(['algo', 'ratio'])

    for key in keys:
        spec = METRIC_SPECS[key]
        plt.figure(figsize=(8.8, 4.8))
        any_line = False

        for algo_name in algo_order:
            sub = df_all[df_all['algo'] == algo_name]
            if sub.empty:
                continue
            stat = agg_mean_ci(sub, key)
            if stat.empty:
                continue

            xs = (stat['ratio'] * 100).values  # 横轴：恶意比例(%)
            ys = stat['mean'].values
            ci = stat['ci95'].values if seeds > 1 else np.zeros_like(ys)

            # 百分比转 %
            ys_plot = ys * 100.0 if spec["percent"] else ys
            ci_plot = ci * (100.0 if spec["percent"] else 1.0)

            plt.plot(xs, ys_plot, marker='o', linewidth=1.8, label=algo_name)
            if seeds > 1:
                plt.fill_between(xs, ys_plot - ci_plot, ys_plot + ci_plot, alpha=0.18)

            any_line = True

        if not any_line:
            plt.close()
            continue

        plt.xlabel('Malicious ratio (%)')
        ylabel = f'{spec["label"]} ({spec["unit"]})' if spec["unit"] else spec["label"]
        plt.ylabel(ylabel)
        plt.title(spec["label"])
        if spec["percent"]:
            plt.ylim(0, 100)
            # 画一条 90% 参考线，方便和 func_life 指标呼应
            if key in ("pdr", "timely_transfer_rate"):
                plt.axhline(90, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        plt.xticks([10,20,30,40,50])
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend(ncol=2, frameon=False)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'line_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'line_{key}.svg'), dpi=150)
        plt.close()


# >>> CHANGED: 容错、网格、线宽、SVG 导出
def plot_timely_curves(tags, out_dir: str, suffix: str = ""):
    plt.figure(figsize=(8, 4.2))
    for tag in tags:
        fname = f'hist_{suffix}_{tag}.csv' if suffix else f'hist_{tag}.csv'
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            print(f"[warn] history file not found: {fpath}")
            continue
        h = pd.read_csv(fpath)
        plt.plot(h['round'], h['cum_timely_rate'], label=tag, linewidth=1.8)
    plt.xlabel('round')
    plt.ylabel('cumulative timely transfer rate')
    plt.legend()
    plt.title('Timely Transfer Rate (Cumulative)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    name = f'timely_rate_curves_{suffix}.png' if suffix else 'timely_rate_curves.png'
    plt.savefig(os.path.join(out_dir, name), dpi=150)
    # 另存 SVG
    name_svg = f'timely_rate_curves_{suffix}.svg' if suffix else 'timely_rate_curves.svg'
    plt.savefig(os.path.join(out_dir, name_svg), dpi=150)
    plt.close()


def parse_ratios(s: str):
    items = [x.strip() for x in s.split(',') if x.strip()]
    vals = []
    for it in items:
        try:
            v = float(it)
            if 0.0 < v < 1.0:
                vals.append(v)
        except:
            pass
    return sorted(set(vals))

# >>> CHANGED: 每个算法用自己的 xs（防止缺某个 ratio 时长度不匹配），误差带对齐
def agg_mean_std(df_all: pd.DataFrame, key: str):
    g = df_all.groupby(['algo', 'ratio'])[key]
    mu = g.mean().reset_index(name='mean')
    sd = g.std().fillna(0.0).reset_index(name='std')
    cnt = g.count().reset_index(name='n')
    out = mu.merge(sd, on=['algo', 'ratio']).merge(cnt, on=['algo', 'ratio'])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices=list(ALGOS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ratios', type=str, help='例如 0.1,0.2,0.3,0.4,0.5')
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--until-dead', action='store_true', help='忽略 --rounds，直至能量耗尽 (LND)')
    parser.add_argument('--seeds', type=int, default=1, help='重复实验次数（不同随机种子）')
    args = parser.parse_args()



    out_dir= 'outputs'; os.makedirs(out_dir, exist_ok=True)
    tags = [args.algo] if args.algo else list(ALGOS.keys())
    if args.all or (not args.algo): tags=list(ALGOS.keys())

    if args.ratios:
        ratios = parse_ratios(args.ratios)
        if not ratios:
            print("ratios 解析失败，请使用例如 --ratios 0.1,0.2,0.3,0.4,0.5")
            sys.exit(1)
        all_rows=[]
        seeds_list=list(range(42, 42+args.seeds))
        for r in ratios:
            n_nodes=100; n_mal=int(n_nodes*r)
            df = run_all_tags(tags, n_nodes, n_mal, seeds_list, rounds=args.rounds,
                              until_dead=args.until_dead, out_dir=out_dir, suffix=f'ratio_{int(r*100)}')
            # 追加 drop_rate 派生列
            df['ratio'] = r
            # >>> CHANGED: 统一端到端口径
            df['drop_rate'] = 1.0 - df['pdr']
            all_rows.append(df)
            plot_timely_curves(tags, out_dir, suffix=f'ratio_{int(r*100)}')
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, 'summary_all.csv'), index=False)
        plot_lines_with_error(df_all, out_dir, seeds=args.seeds)
        print("批量完成：summary_all.csv 与 line_*.png 已生成。")
    else:
        seeds_list=list(range(42, 42+args.seeds))
        df = run_all_tags(tags, 100, 30, seeds_list, rounds=args.rounds,
                          until_dead=args.until_dead, out_dir=out_dir, suffix="")
        # >>> CHANGED: 统一端到端口径
        df['drop_rate'] = 1.0 - df['pdr']
        plot_bars(df, out_dir)
        plot_timely_curves(tags, out_dir)
        print("单次运行完成：summary.csv / metric_*.png / timely_rate_curves.png")

if __name__ == '__main__':
    main()
