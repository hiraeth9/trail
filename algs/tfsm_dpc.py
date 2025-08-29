# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Optional
import math, random
import numpy as np

from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster, dist, e_tx, e_rx, clamp,
    BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS
)

class TFSM_DPC(AlgorithmBase):
    """
    高效版（论文一致 + 性能修复）
    - DT：式(5)(6)(7)，评估时用 eval 版（不写回）；真正写回仅在 apply_watchdog
    - DPC：式(12)(13)(14) + K=3 + 上/下界基线；按 (i,j,commons) 缓存一次
    - IT/CT：式(8)(9)(10)(11)，IT 含相异度过滤；所有 pair 结果按轮缓存
    - 邻居/共同邻居：九宫格网格索引 + 按需缓存，避免 O(N^2)
    """

    # §3.1 式(5)(6)(7)
    lam = 0.85
    a2  = 6.0

    # §3.3 Table 2
    eta = 0.50        # IT 权重
    CT_th = 0.35      # 黑名单阈值（综合信任）

    # DPC 设置
    K = 3
    bench_low, bench_high = 0.15, 0.85
    dc_percent = 0.02  # 距离分位选 dc

    # 框架信任/黑名单
    forget = 0.98
    strike_threshold = 3
    trust_black_fallback = 0.20

    @property
    def name(self): return "TFSM-DPC-2024"

    @property
    def trust_blacklist(self) -> float:
        return self.CT_th

    @property
    def trust_warn(self) -> float:
        return max(0.5, self.CT_th + 0.2)

    # ========= 轮级缓存与网格索引 =========
    def _reset_round_cache(self):
        self._rc_round = -1
        self._grid = {}
        self._pos_cache = {}
        self._cell_size = COMM_RANGE
        self._neigh: Dict[int, List[Node]] = {}
        self._commons: Dict[Tuple[int,int], List[Node]] = {}
        self._det_cache: Dict[int, float] = {}
        self._dt_eval_cache: Dict[Tuple[int,int], float] = {}
        self._it_pack_cache: Dict[Tuple[int,int,Tuple[int,...]], Tuple[Optional[float], Tuple[int,...]]] = {}
        self._ct_cache: Dict[Tuple[int,int], float] = {}
        self._nid2node = {}  # <--- 新增：本轮 nid→Node 映射

    def _build_grid(self):
        S = self.sim.alive_nodes()
        cell = self._cell_size
        self._grid.clear(); self._pos_cache.clear()
        for n in S:
            x, y = n.pos()
            cx, cy = int(x // cell), int(y // cell)
            self._grid.setdefault((cx, cy), []).append(n)
            self._pos_cache[n.nid] = (x, y)

    def _nearby_from_grid(self, node: Node, radius: float) -> List[Node]:
        if not self._pos_cache:
            self._build_grid()
        x, y = self._pos_cache.get(node.nid, node.pos())
        cx, cy = int(x // self._cell_size), int(y // self._cell_size)
        out = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for m in self._grid.get((cx+dx, cy+dy), []):
                    if m.nid == node.nid: continue
                    if dist((x, y), m.pos()) <= radius:
                        out.append(m)
        return out

    def _ensure_round_cache(self):
        if getattr(self, "_rc_round", None) == self.sim.round:
            return
        S = self.sim.alive_nodes()
        self._nid2node = {n.nid: n for n in S}  # <--- 新增：本轮 nid→Node
        self._det_cache = {n.nid: (1.0 if n.energy >= 0.2 else 0.0) for n in S}  # DET 的能量阈值 0.2J
        self._dt_eval_cache.clear(); self._it_pack_cache.clear(); self._ct_cache.clear()
        self._neigh.clear(); self._commons.clear()
        self._build_grid()
        self._rc_round = self.sim.round

    def _neighbors_cached(self, x: Node) -> List[Node]:
        self._ensure_round_cache()
        if x.nid not in self._neigh:
            self._neigh[x.nid] = [m for m in self._nearby_from_grid(x, COMM_RANGE) if m.alive and not m.blacklisted]
        return self._neigh[x.nid]

    def _commons_cached(self, i: Node, j: Node) -> List[Node]:
        self._ensure_round_cache()
        key = (i.nid, j.nid)
        if key in self._commons: return self._commons[key]
        Ni = self._neighbors_cached(i)
        Nj_ids = {n.nid for n in self._neighbors_cached(j)}
        commons = [n for n in Ni if n.nid in Nj_ids]
        self._commons[key] = commons
        return commons

    # ========= 直信任（eval 版 & 写回版） =========
    def _ensure_edge_state(self, u: Node, v: Node):
        if not hasattr(u, "dt_alpha"): u.dt_alpha = {}
        if not hasattr(u, "dt_beta"):  u.dt_beta  = {}
        if not hasattr(u, "dt_n"):     u.dt_n     = {}
        if not hasattr(u, "dt_m"):     u.dt_m     = {}
        for d in (u.dt_alpha, u.dt_beta, u.dt_n, u.dt_m):
            d.setdefault(v.nid, 0.0)

    def _theta_penalty(self, AC: float) -> float:
        return 1.5 - 1.0 / (1.0 + math.exp(-self.a2 * clamp(AC, 0.0, 1.0)))

    def _direct_trust_eval(self, i: Node, j: Node) -> float:
        """不写回，仅根据当前 α/β 与本轮 n/m 计算 DT_t"""
        self._ensure_edge_state(i, j)
        a_prev, b_prev = i.dt_alpha[j.nid], i.dt_beta[j.nid]
        n_t, m_t = i.dt_n[j.nid], i.dt_m[j.nid]
        a_t = self.lam * a_prev + n_t
        b_t = self.lam * b_prev + m_t
        AC = 0.0 if (n_t + m_t) <= 1e-9 else (m_t / (n_t + m_t))
        theta = self._theta_penalty(AC)
        return clamp((a_t + 1.0) / (a_t + b_t + 2.0 * theta + 1e-12), 0.0, 1.0)

    def _direct_trust_DT(self, i: Node, j: Node) -> float:
        """写回版：仅在 apply_watchdog/最终融合场景调用"""
        dt = self._direct_trust_eval(i, j)
        # 将 a_t/b_t 写回，作为下一轮历史
        a_prev, b_prev = i.dt_alpha[j.nid], i.dt_beta[j.nid]
        n_t, m_t = i.dt_n[j.nid], i.dt_m[j.nid]
        i.dt_alpha[j.nid] = self.lam * a_prev + n_t
        i.dt_beta[j.nid]  = self.lam * b_prev + m_t
        return dt

    def _dt_eval_cached(self, i: Node, j: Node) -> float:
        key = (i.nid, j.nid)
        if key not in self._dt_eval_cache:
            self._dt_eval_cache[key] = self._direct_trust_eval(i, j)
        return self._dt_eval_cache[key]

    # ========= DPC =========
    def _pairwise_dist(self, pts: np.ndarray) -> np.ndarray:
        X = pts.astype(float)  # (N,2)
        d2 = np.sum((X[:, None, :] - X[None, :, :])**2, axis=2)
        return np.sqrt(d2 + 1e-12)

    def _dc_from_dist(self, D: np.ndarray) -> float:
        vals = D[np.triu_indices(D.shape[0], 1)]
        vals.sort()
        k = max(1, int(self.dc_percent * len(vals)))
        return float(vals[k-1])

    def _dpc_cluster(self, pts_xy: np.ndarray, K: int) -> Tuple[np.ndarray, List[int]]:
        N = pts_xy.shape[0]
        D = self._pairwise_dist(pts_xy)
        dc = self._dc_from_dist(D)

        # rho
        rho = np.exp(-(D**2) / (dc**2 + 1e-12)).sum(axis=1) - 1.0  # 去掉自身
        # delta
        order = np.argsort(-rho)
        delta = np.zeros(N)
        for idx, x in enumerate(order):
            if idx == 0:
                delta[x] = float(np.max(D[x, :]))
            else:
                higher = order[:idx]
                delta[x] = float(np.min(D[x, higher]))
        phi = rho * delta
        centers = list(np.argsort(-phi)[:K])

        labels = -np.ones(N, dtype=int)
        for c_i, c in enumerate(centers):
            labels[c] = c_i
        for x in order:
            if labels[x] != -1: continue
            higher = [y for y in range(N) if rho[y] > rho[x]]
            if not higher:
                labels[x] = labels[centers[0]]
                continue
            y_best = min(higher, key=lambda y: D[x, y])
            labels[x] = labels[y_best]
        return labels, centers

    # ========= IT：DPC 过滤 + 权重 =========
    def _filter_and_compute_IT(self, i: Node, j: Node, commons: List[Node]) -> Tuple[Optional[float], List[Node]]:
        if not commons:
            return None, []

        # 缓存键（和一轮绑定）：(i, j, sorted commons)
        key = (i.nid, j.nid, tuple(sorted(n.nid for n in commons)))
        if key in self._it_pack_cache:
            IT, falsers_ids = self._it_pack_cache[key]
            # 用轮级映射把 id 变回 Node；若个别 id 不在映射里就跳过
            return IT, [self._nid2node[fid] for fid in falsers_ids if fid in self._nid2node]

        # 构造 (x, DT_kj) + 两个基线点
        pts = []
        id_map = []
        xidx = 1.0
        for k in commons:
            dt_kj = self._dt_eval_cached(k, j)
            pts.append([xidx, float(dt_kj)])
            id_map.append(k)
            xidx += 1.0
        pts.append([0.0,  self.bench_low])
        pts.append([1e6, self.bench_high])
        pts_xy = np.array(pts, dtype=float)

        labels, centers = self._dpc_cluster(pts_xy, self.K)
        lbl_low  = labels[len(id_map)]
        lbl_high = labels[len(id_map)+1]

        trusted, falsers = [], []
        for idx, k in enumerate(id_map):
            if labels[idx] == lbl_low or labels[idx] == lbl_high:
                falsers.append(k)
            else:
                trusted.append(k)

        if not trusted:
            self._it_pack_cache[key] = (None, tuple(n.nid for n in falsers))
            return None, falsers

        # 权重：式(9) —— 用 DT_ik 归一
        w_den = 0.0
        w = {}
        for k in trusted:
            DT_ik = self._dt_eval_cached(i, k)
            w[k.nid] = DT_ik
            w_den += DT_ik
        if w_den <= 1e-12:
            self._it_pack_cache[key] = (None, tuple(n.nid for n in falsers))
            return None, falsers

        IT = 0.0
        for k in trusted:
            DT_ik = w[k.nid] / w_den
            DT_kj = self._dt_eval_cached(k, j)
            IT += DT_ik * (w[k.nid]) * DT_kj  # 式(8)：wk * DT_ik * DT_kj

        IT = clamp(IT, 0.0, 1.0)
        self._it_pack_cache[key] = (IT, tuple(n.nid for n in falsers))
        return IT, falsers

    def _comprehensive_trust_CT(self, i: Node, j: Node) -> Tuple[float, List[Node]]:
        self._ensure_round_cache()
        key = (i.nid, j.nid)
        if key in self._ct_cache:
            val = self._ct_cache[key]
            # falsers 对 finalize 影响小，这里不复用（避免大对象存储）
            commons = self._commons_cached(i, j)
            _, falsers = self._filter_and_compute_IT(i, j, commons)
            return val, falsers

        DT = self._dt_eval_cached(i, j)
        commons = self._commons_cached(i, j)
        IT, falsers = self._filter_and_compute_IT(i, j, commons)
        if IT is None:
            CT = DT  # 式(11)
        else:
            CT = (1.0 - self.eta) * DT + self.eta * IT  # 式(10)
        CT = clamp(CT, 0.0, 1.0)
        self._ct_cache[key] = CT
        return CT, falsers

    # ========= 五个框架钩子 =========
    def select_cluster_heads(self):
        """
        论文对 CH 选举未约束；保留轻量随机 + 冷却。为了性能，先清一轮缓存。
        """
        self._reset_round_cache()
        sim = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN:
                continue
            if random.random() < BASE_P_CH:
                n.is_ch = True
                n.last_ch_round = sim.round
                sim.clusters[n.nid] = Cluster(n.nid)
        if len(sim.clusters) == 0:
            pick = random.choice(alive)
            pick.is_ch = True; pick.last_ch_round = sim.round
            sim.clusters[pick.nid] = Cluster(pick.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        return False

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        从网格近邻内选 CT 最大的 CH 作为下一跳；无则直达。
        """
        self._ensure_round_cache()
        sim = self.sim
        d_bs = dist(ch.pos(), sim.bs)
        direct_cost = e_tx(DATA_PACKET_BITS, d_bs)

        local = self._nearby_from_grid(ch, CH_NEIGHBOR_RANGE)
        cands = [x for x in local if x.alive and (not x.blacklisted) and x.nid != ch.nid]
        if not cands:
            ch.last_selected_relay = None
            return None, {}

        scored = []
        for nb in cands:
            ct, _ = self._comprehensive_trust_CT(ch, nb)
            d1 = dist(ch.pos(), nb.pos())
            d2 = dist(nb.pos(), sim.bs)
            twohop = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)
            scored.append((ct, nb, twohop))
        scored.sort(key=lambda t: (t[0], -t[2]))
        best_ct, relay, cost2 = scored[0]
        ch.last_selected_relay = relay
        return relay, {'ct': best_ct, 'cost2': cost2, 'cost1': direct_cost}

    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        只在这里更新直信任的计数（n_t/m_t），并做节点级观测；评估阶段不写状态。
        """
        j = getattr(ch, "last_selected_relay", None)
        if j is not None:
            self._ensure_edge_state(ch, j)
            if ok and timely:
                ch.dt_n[j.nid] = ch.dt_n.get(j.nid, 0.0) + 1.0
            else:
                ch.dt_m[j.nid] = ch.dt_m.get(j.nid, 0.0) + 1.0
        # 节点级观测
        if ok and timely:
            ch.observed_success += 1.0
            ch.consecutive_strikes = max(0, ch.consecutive_strikes - 1)
        else:
            ch.observed_fail += 1.0
            ch.consecutive_strikes += 1

    def finalize_trust_blacklist(self):
        """
        轮末：
        - 节点级信任计数做指数遗忘；
        - 把“虚假推荐者”并入 β 通道（快速下沉恶意）；
        - 基于邻域 CT 中位数进行鲁棒黑名单判别；
        - 清零本轮 n/m 计数，等待下一轮。
        """
        self._ensure_round_cache()
        S = self.sim.alive_nodes()

        # 1) 节点级遗忘 & 击穿保护
        for n in S:
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail
            if n.consecutive_strikes >= self.strike_threshold:
                n.blacklisted = True

        # 2) 将 DPC 识别的虚假推荐者计为异常（β 通道）
        for i in S:
            Ni = self._neighbors_cached(i)
            for j in Ni:
                _, falsers = self._comprehensive_trust_CT(i, j)
                for k in falsers:
                    self._ensure_edge_state(i, k)
                    i.dt_m[k.nid] = i.dt_m.get(k.nid, 0.0) + 1.0

        # 3) 邻域 CT 中位数做黑名单裁决
        for j in S:
            Ni = self._neighbors_cached(j)
            if not Ni:
                continue
            vals = [self._comprehensive_trust_CT(i, j)[0] for i in Ni]
            med_ct = float(np.median(vals)) if vals else 1.0
            if med_ct < self.CT_th or (med_ct < self.trust_black_fallback and j.consecutive_strikes >= 1):
                j.blacklisted = True

        # 4) 将 α/β 写回并清零本轮 n/m（用于下一轮）
        for i in S:
            if hasattr(i, "dt_alpha"):
                for nid in list(i.dt_alpha.keys()):
                    j = self._nid2node.get(nid)  # ← 用轮级映射取 Node
                    if j is not None:
                        self._direct_trust_DT(i, j)
            if hasattr(i, "dt_n"):
                for nid in list(i.dt_n.keys()):
                    i.dt_n[nid] = 0.0
            if hasattr(i, "dt_m"):
                for nid in list(i.dt_m.keys()):
                    i.dt_m[nid] = 0.0

