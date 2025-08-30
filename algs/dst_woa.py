# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Optional
import random, math
import numpy as np

# 这些符号来自你的核心框架；如命名略有差异，请按工程替换
from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster, dist, e_tx, e_rx, clamp,
    COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS, CH_COOLDOWN
)

class DST_WOA(AlgorithmBase):
    """
    论文一致版本：DST-WOA
    Phase-1: 基于 Dempster-Shafer 证据合成的 CH 选择
    Phase-2: CH Recycling (min_erg, tl)
    Phase-3: Whale Optimization Algorithm 选择最佳 CH→…→CH→BS 路径
      Fitness = E + 1/D + 1/H + 1/Scdis   [Eq.(11)]
      Scdis   = avg link distance along route [Eq.(12)]
    参考：论文第 4 节伪代码与式(11)(12)；图 1 流程图。
    """

    # ========= 可调参数（与论文一致/对齐） =========
    # Phase-2：CH 回收阈值
    min_erg_ratio = 0.60      # min_erg = 0.6（论文）——采用“相对初始”近似：使用全网中位能量比例
    ttl_rounds    = 60        # tl = 60s；仿真每轮近似1秒时与论文等价；如非1s步长，请据实调整

    # WOA 参数（Phase-3）
    WOA_POP   = 24
    WOA_ITERS = 30
    WOA_b     = 1.0           # 螺旋常数
    WOA_max_hops = 8          # 限制 CH→BS 最大跳数（避免搜索爆炸）

    # 兼容性保护（主程序信任/拉黑阈值）
    trust_black = 0.35
    forget      = 0.98
    strike_threshold = 3

    @property
    def name(self): return "DST-WOA-2024"

    @property
    def trust_blacklist(self) -> float:
        # 用我们在类里配置的黑名单阈值（之前用变量名 trust_black）
        return getattr(self, "trust_black", 0.35)

    @property
    def trust_warn(self) -> float:
        # 可选的预警阈值；部分框架/工具代码可能会读取
        return max(0.5, self.trust_blacklist + 0.2)

    # ========== Phase-1：DST 信任与 CH 选择 ==========

    def _two_observers(self, x: Node, alive: List[Node]) -> List[Node]:
        """选取两个“就近且可信”的邻居作为观察者（论文：两个邻居给出证据）"""
        neigh = []
        for n in alive:
            if n.nid == x.nid or (not n.alive) or n.blacklisted: continue
            d = dist(n.pos(), x.pos())
            if d <= COMM_RANGE:
                neigh.append((d, n.trust(), n))
        # 距离近且 trust 高者优先
        neigh.sort(key=lambda t: (t[0], -t[1]))
        out = [n for _,__,n in neigh[:2]]
        if len(out) < 2 and neigh:
            out = [neigh[0][2]] * 2  # 退化情况：只有1个邻居时重复使用
        return out

    def _opinion_of(self, observer: Node, target: Node) -> float:
        """
        观察者对 target 的可信率 l \in [0,1]。
        工程上若维护了邻居级意见，可从 observer.opinions[target.nid] 取用；
        否则采用 sqrt(observer.trust() * target.trust()) 的保守合成。
        """
        if hasattr(observer, "opinions") and isinstance(observer.opinions, dict):
            if target.nid in observer.opinions:
                return clamp(float(observer.opinions[target.nid]), 0.0, 1.0)
        return clamp(math.sqrt(observer.trust() * target.trust()), 0.0, 1.0)

    def _dst_combine_two(self, l1: float, l2: float) -> float:
        """
        Dempster 组合：Ω={T,F}，两个邻居给出 A1(Y)=l1, A1(Y')=1-l1；A2 同理。
        论文式(7)–(10) 的二元特例，实现冲突归一化 W 并返回对 {T} 的合成置信。 [Phase-1]
        """
        a1_t, a1_f = l1, (1.0 - l1)
        a2_t, a2_f = l2, (1.0 - l2)
        # 冲突项 K = A1(T)A2(F) + A1(F)A2(T)
        K = a1_t * a2_f + a1_f * a2_t
        W = 1.0 - K  # 论文中归一化因子 W；二元情形 W = 1-K
        if W <= 1e-12:
            return 0.5  # 完全冲突→不确定
        # 合成对 {T} 的质量
        mT = (a1_t * a2_t + a1_t * (0) + (0) * a2_t) / W  # 只剩交集 T∩T
        # 二元特例下即 mT = (a1_t * a2_t) / W
        return clamp(mT, 0.0, 1.0)

    def _dst_trust_of_node(self, x: Node, alive: List[Node]) -> float:
        """候选节点 x 的 DST 合成可信率（两个邻居给出 l1, l2 后按上式合成）"""
        obs = self._two_observers(x, alive)
        if len(obs) < 2:
            # 无足够邻居：退化为自身 trust
            return x.trust()
        l1 = self._opinion_of(obs[0], x)
        l2 = self._opinion_of(obs[1], x)
        return self._dst_combine_two(l1, l2)

    def select_cluster_heads(self):
        """
        Phase-1：按论文伪代码：
          - 计算每个节点的 DST 信任合成值
          - 以合成值从高到低挑选；并考虑 cooldown/回收（Phase-2 的 tl/min_erg）
        """
        sim = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return

        # 动态能量基线（用于回收判据）：用存活节点能量中位数近似“初始能量比例”
        med_e = float(np.median([n.energy for n in alive]))

        scored: List[Tuple[float, Node]] = []
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN:
                continue
            # Phase-2: 回收判据（min_erg/tl）提前过滤
            too_old = (sim.round - n.last_ch_round) >= self.ttl_rounds and getattr(n, "is_ch", False)
            too_low = (n.energy <= self.min_erg_ratio * max(1e-9, med_e)) and getattr(n, "is_ch", False)
            if too_old or too_low:
                n.is_ch = False  # 触发回收：不允许继续担任
                continue

            cs = self._dst_trust_of_node(n, alive)  # DST 合成置信
            scored.append((cs, n))

        # 以 DST 置信降序选择前 8% 作为 CH（增加簇头数量）
        scored.sort(key=lambda t: t[0], reverse=True)
        need_total = max(1, int(0.08 * len(alive)))
        sim.clusters.clear()
        for _, pick in scored[:need_total]:
            pick.is_ch = True
            pick.last_ch_round = sim.round
            sim.clusters[pick.nid] = Cluster(pick.nid)

    # ========== Phase-2：CH Recycling 的运行时检查 ==========
    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        按论文 Phase-2：若 CH 电量跌破 min_erg 或超出 tl（round 近似秒），则“需要回收”；
        这里以累计击穿形式记录，统一在 finalize 中处理（与框架风格一致）。
        """
        sim = self.sim
        alive = sim.alive_nodes()
        med_e = float(np.median([n.energy for n in alive])) if alive else 0.0

        energy_low = (ch.energy <= self.min_erg_ratio * max(1e-9, med_e))
        time_over  = ((sim.round - ch.last_ch_round) >= self.ttl_rounds)

        breach = energy_low or time_over or (not ok) or (not timely)
        if breach:
            ch.observed_fail += 1.0
            ch.consecutive_strikes += 1
            # 标记建议回收
            ch.to_recycle = True
        else:
            ch.observed_success += 1.0
            ch.consecutive_strikes = max(0, ch.consecutive_strikes - 1)
            ch.to_recycle = False

    def finalize_trust_blacklist(self):
        """指数遗忘更新信任；满足低信任+多次击穿则拉黑；执行回收。"""
        for n in self.sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail
            if getattr(n, "to_recycle", False):
                n.is_ch = False  # 执行回收
            if (n.trust() < self.trust_black) and (n.consecutive_strikes >= 1 or n.observed_fail > n.observed_success):
                n.blacklisted = True
            if n.consecutive_strikes >= self.strike_threshold:
                n.blacklisted = True

    # ========= Phase-3：WOA 路由 =========

    # ------ 路由适应度：论文式(11)(12) ------
    def _route_metrics(self, route: List[Node]) -> Tuple[float, float, int, float]:
        """
        route: [CH0=src, CH1, ..., CHk]  —— 最后从 CHk 直达 BS
        返回 (E_norm, D_total, H, Scdis)
          E_norm : 路径上 CH 节点能量的均值（按全局 CH 能量最大值归一）
          D_total: CH 间总距离（加上 CHk→BS）
          H      : 中继跳数（CH 间跳数 + 最后 CH→BS 视作 1 跳）
          Scdis  : (D_total / H) —— 即平均链路距离（符合式(12)的平均）
        """
        if not route:
            return 0.0, 1e9, 999, 1e9
        sim = self.sim
        e_max = max([n.energy for n in route]) if route else 1.0
        e_norm = sum(n.energy for n in route) / (len(route) * (e_max + 1e-9))

        # CH 间距离
        d_sum = 0.0
        for i in range(1, len(route)):
            d_sum += dist(route[i-1].pos(), route[i].pos())
        # 最后 CH -> BS
        d_sum += dist(route[-1].pos(), sim.bs)

        # 跳数：CH 间边数 + 1（最后到BS）
        H = (len(route) - 1) + 1
        scdis = d_sum / (H + 1e-9)
        return e_norm, d_sum, H, scdis

    def _fitness(self, route: List[Node]) -> float:
        """
        论文式(11)：Fitness = E + 1/D + 1/H + 1/Scdis
        注意：为数值稳定做了极小值保护。
        """
        E, D, H, Sc = self._route_metrics(route)
        return (E +
                1.0 / (D + 1e-6) +
                1.0 / (H + 1e-6) +
                1.0 / (Sc + 1e-6))

    # ------ 路由搜索基本图操作 ------
    def _ch_neighbors(self, node: Node, ch_nodes: List[Node]) -> List[Node]:
        out = []
        for other in ch_nodes:
            if other.nid == node.nid or (not other.alive): continue
            if dist(node.pos(), other.pos()) <= CH_NEIGHBOR_RANGE:
                out.append(other)
        # 近者优先
        out.sort(key=lambda n: dist(node.pos(), n.pos()))
        return out

    # ------ 初始解：若干条可行路径（贪心/BFS 混合） ------
    def _seed_routes(self, src: Node, ch_nodes: List[Node]) -> List[List[Node]]:
        sim = self.sim
        seeds: List[List[Node]] = []

        # 贪心：每步选向 BS 更近的邻居
        for _ in range(3):
            path = [src]
            curr = src
            visited = {src.nid}
            for __ in range(self.WOA_max_hops - 1):
                neigh = self._ch_neighbors(curr, ch_nodes)
                if not neigh: break
                # 选“到BS更近 且 沟通距离更短 的”前若干个候选
                neigh.sort(key=lambda n: (dist(n.pos(), sim.bs), dist(curr.pos(), n.pos())))
                picked = None
                for cand in neigh[:5]:
                    if cand.nid not in visited:
                        picked = cand; break
                if picked is None: break
                path.append(picked); visited.add(picked.nid); curr = picked
                # 看是否直接到 BS 足够近（常量阈值：直接传比继续两跳更省）
                if dist(curr.pos(), sim.bs) <= CH_NEIGHBOR_RANGE:
                    break
            seeds.append(path)

        # BFS 限深若干条
        from collections import deque
        q = deque()
        q.append([src])
        seen = set([src.nid])
        cnt = 0
        while q and cnt < (self.WOA_POP - len(seeds)):
            p = q.popleft()
            curr = p[-1]
            if len(p) >= self.WOA_max_hops:
                seeds.append(p); cnt += 1; continue
            for nb in self._ch_neighbors(curr, ch_nodes)[:6]:
                if nb.nid in set(n.nid for n in p):  # 避免环
                    continue
                npth = p + [nb]
                q.append(npth)
                if dist(nb.pos(), self.sim.bs) <= CH_NEIGHBOR_RANGE:
                    seeds.append(npth); cnt += 1
                    if cnt >= (self.WOA_POP - len(seeds)): break
        # 兜底：至少一条自身直达（若可）
        if not seeds:
            seeds = [[src]]
        return seeds[:self.WOA_POP]

    # ------ 离散 WOA 更新算子：围捕/螺旋/探索 ------
    def _towards_best(self, curr: List[Node], best: List[Node], ch_nodes: List[Node]) -> List[Node]:
        """Encircling prey（离散化）：按位置对齐，局部替换为 best 的对应节点并重新连通"""
        if not best: return curr
        res = [curr[0]]
        i = 1
        used = {curr[0].nid}
        while i < min(len(curr), len(best)):
            pick = best[i]
            if pick.nid not in used and dist(res[-1].pos(), pick.pos()) <= CH_NEIGHBOR_RANGE:
                res.append(pick); used.add(pick.nid)
            else:
                # 就近可达替代
                neigh = self._ch_neighbors(res[-1], ch_nodes)
                alt = next((n for n in neigh if n.nid not in used), None)
                if alt: res.append(alt); used.add(alt.nid)
                else: break
            i += 1
        return res

    def _spiral_attack(self, curr: List[Node], best: List[Node], lval: float, ch_nodes: List[Node]) -> List[Node]:
        """Bubble-net（螺旋，离散化）：对路径做 2-opt/段替换，趋向更短/更近 BS"""
        if len(curr) < 3: return curr
        a = random.randint(1, len(curr) - 2)
        b = random.randint(a, len(curr) - 1)
        new = curr[:a] + list(reversed(curr[a:b])) + curr[b:]
        # 若断边不可达则回退
        ok = True
        for i in range(1, len(new)):
            if dist(new[i-1].pos(), new[i].pos()) > CH_NEIGHBOR_RANGE:
                ok = False; break
        return new if ok and self._fitness(new) >= self._fitness(curr) else curr

    def _random_explore(self, curr: List[Node], ch_nodes: List[Node]) -> List[Node]:
        """Exploration：随机在某处插入/替换一个可达邻居"""
        base = curr[:]
        anchor_idx = random.randint(0, len(base)-1)
        anchor = base[anchor_idx]
        neigh = self._ch_neighbors(anchor, ch_nodes)
        if not neigh: return base
        cand = random.choice(neigh)
        if cand.nid in [n.nid for n in base]: return base
        # 尝试把 cand 插到 anchor 后
        new = base[:anchor_idx+1] + [cand] + base[anchor_idx+1:]
        # 可达性检查
        ok = True
        for i in range(1, len(new)):
            if dist(new[i-1].pos(), new[i].pos()) > CH_NEIGHBOR_RANGE:
                ok = False; break
        return new if ok else base

    def _woa_best_route(self, src: Node, ch_nodes: List[Node]) -> List[Node]:
        """WOA 主过程：返回从 src 到 BS 的最佳 CH 路径（不含 BS）"""
        pop = self._seed_routes(src, ch_nodes)
        if not pop: pop = [[src]]
        # 记录最好个体
        best = max(pop, key=self._fitness)

        # 标准 WOA：a 从 2 线性降到 0；p ∈ [0,1] 决定 bubble-net 或 encircle/search
        for t in range(self.WOA_ITERS):
            a = 2 - 2 * (t / max(1, self.WOA_ITERS - 1))
            new_pop = []
            for r in pop:
                p = random.random()
                A = 2 * a * random.random() - a
                C = 2 * random.random()
                if p < 0.5:
                    if abs(A) < 1:
                        # encircling prey
                        nr = self._towards_best(r, best, ch_nodes)
                    else:
                        # search for prey（远离随机个体）
                        rand_i = random.randrange(len(pop))
                        nr = self._random_explore(pop[rand_i], ch_nodes)
                else:
                    # bubble-net (spiral)
                    lval = random.uniform(-1, 1)
                    nr = self._spiral_attack(r, best, lval, ch_nodes)
                new_pop.append(nr)

            # 精英保留 + 更新最优
            new_best = max(new_pop + [best], key=self._fitness)
            best = new_best
            pop = new_pop

        return best

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        # 论文未给出成员侧冗余转发，保持关闭
        return False

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        Phase-3：利用 WOA 在 CH 图上找“全局最佳路径”，再取下一跳为该路径的第二个节点；
        若路径长度为 1（仅自己），则直达 BS。
        返回：(relay_node | None, debug_info)
        """
        if not ch_nodes or len(ch_nodes) == 1:
            return None, {}

        # 找最佳路径
        best_route = self._woa_best_route(ch, ch_nodes)

        # 计算能耗（两跳与直达对比，仅做调试信息，不做硬约束；论文适应度已包含 D/H/Scdis）
        sim = self.sim
        d_bs = dist(ch.pos(), sim.bs)
        cost_direct = e_tx(DATA_PACKET_BITS, d_bs)

        relay = None
        if len(best_route) >= 2 and best_route[0].nid == ch.nid:
            relay = best_route[1]

        info = {}
        if relay is not None:
            d1 = dist(ch.pos(), relay.pos())
            d2 = dist(relay.pos(), sim.bs)
            cost_twohop = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)
            info = {'route_len': len(best_route), 'cost1': cost_direct, 'cost2': cost_twohop}
        return relay, info
