
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
def plot_bars(summary: pd.DataFrame, out_dir: str, algo_order=None):
    if algo_order is None:
        algo_order = ["SHEER-2025", "ACTAR-2024", "TFSM-DPC-2024", "DST-WOA-2024", "TRAIL (ours)"]

    metrics = [
        ('pdr', 'Packet Delivery Ratio'),
        ('timely_transfer_rate', 'Timely Transfer Rate'),
        ('energy_per_delivered', 'Energy per Delivered (J/packet)'),
        ('throughput_bits_per_round', 'Throughput (bits/round)'),
        ('avg_hops_to_bs', 'Average Hops to BS'),
        ('drop_p', 'Total Drops'),
        ('FND', 'FND'),
        ('HND', 'HND'),
        ('LND', 'LND'),
    ]

    # 预聚合（按 algo ）
    grouped = summary.groupby('algo')
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(numeric_only=True).fillna(0.0)
    counts = grouped.size()

    # 算法显示顺序（只保留出现过的）
    algos = [a for a in algo_order if a in means.index]
    x = np.arange(len(algos))

    for key, title in metrics:
        if key not in means.columns:
            continue
        y = means.loc[algos, key].values
        yerr = stds.loc[algos, key].values if (counts.max() > 1) else None

        plt.figure(figsize=(8, 4.2))
        bars = plt.bar(x, y, yerr=yerr, capsize=4)
        plt.xticks(x, algos, rotation=20, ha='right')
        plt.title(title)
        plt.ylabel(key)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # 数值标注
        for i, b in enumerate(bars):
            val = y[i]
            plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                     f'{val:.3g}' if abs(val) < 1000 else f'{val:,.0f}',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'metric_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'metric_{key}.svg'), dpi=150)  # 矢量图
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

def plot_lines_with_error(df_all: pd.DataFrame, out_dir: str, seeds: int):
    algo_order = ["SHEER-2025", "ACTAR-2024", "TFSM-DPC-2024", "DST-WOA-2024", "TRAIL (ours)"]
    metrics = [
        ('pdr', 'PDR (Packet Delivery Ratio)'),
        ('drop_rate', 'Drop Rate'),
        ('timely_transfer_rate', 'Timely Transfer Rate'),
        ('energy_per_delivered', 'Energy per Delivered Packet (J/packet)'),
        ('throughput_bits_per_round', 'Throughput (bits/round)'),
        ('avg_hops_to_bs', 'Average Hops to BS'),
        ('FND', 'FND (First Node Dies)'),
        ('HND', 'HND (Half Nodes Die)'),
        ('LND', 'LND (Last Node Dies)'),
        ('func_life_pdr90', 'Functional Lifetime (PDR≥0.90)'),
    ]

    for key, title in metrics:
        plt.figure(figsize=(8.6, 4.6))
        for algo_name in algo_order:
            sub = df_all[df_all['algo'] == algo_name]
            if sub.empty:
                continue
            stats = agg_mean_std(sub, key).sort_values('ratio')
            if stats.empty:
                continue
            xs = (stats['ratio'] * 100).values   # 每个算法自己的 x（百分比）
            ys = stats['mean'].values
            plt.plot(xs, ys, marker='o', linewidth=1.8, label=algo_name)
            if seeds > 1:
                sd = stats['std'].values
                plt.fill_between(xs, ys - sd, ys + sd, alpha=0.18)

        plt.xlabel('Malicious ratio (%)')
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'line_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'line_{key}.svg'), dpi=150)
        plt.close()


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
