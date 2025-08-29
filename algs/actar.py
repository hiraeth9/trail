# -*- coding: utf-8 -*-
from typing import List, Tuple, Dict, Optional
import math, random
import numpy as np

from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster, dist, e_tx, e_rx, clamp,
    BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS
)

class ACTAR(AlgorithmBase):
    """
    高效版（论文一致 + 性能修复）
    - AHC：近域单簇 + 远域若干簇
    - MOCH：MOF = w_cov*Cov + w_e*ERe + w_cost*Ccost + w_prox*Pn
    - TAR：TD = v1*(wB*DBT + wE*DET) + v2*(uB*IBT' + uE*DET)，IBT' 含相异度过滤
    关键优化：
      * 评估函数无副作用（不在评估时做时衰/写状态）
      * 每轮构建邻居与共同邻居缓存、DET/DT/IT/TD 缓存
    """

    # ====== AHC ======
    pn = 0.20
    ndc_min = 1
    ndc_max_ratio = 0.15

    # ====== MOCH 权重（和=1） ======
    w_cov  = 0.30
    w_e    = 0.30
    w_cost = 0.20
    w_prox = 0.20

    # ====== TAR 权重/阈值 ======
    wB = 0.65  # DBT
    wE = 0.35  # DET
    uB = 0.70  # IBT
    uE = 0.30  # DET
    v1 = 0.55  # DT
    v2 = 0.45  # IT

    rho_pos = 0.20   # 正交互衰减率（只在 apply_watchdog 中使用）
    rho_neg = 0.05   # 负交互衰减率
    Eth = 0.20       # 能量阈值（J）
    delta = 0.25     # 相异度阈值（用于过滤推荐）

    # 框架信任/黑名单
    forget = 0.98
    strike_threshold = 3
    td_blacklist = 0.30

    @property
    def name(self): return "ACTAR-2024"

    @property
    def trust_blacklist(self) -> float:
        return self.td_blacklist

    @property
    def trust_warn(self) -> float:
        return max(0.5, self.td_blacklist + 0.2)

    # ---------------------- 每轮缓存 ----------------------
    def _reset_round_cache(self):
        self._rc_round = -1
        self._neigh: Dict[int, List[Node]] = {}
        self._commons: Dict[Tuple[int,int], List[Node]] = {}
        self._det_cache: Dict[int, float] = {}
        self._dt_cache: Dict[Tuple[int,int], float] = {}
        self._ibt_cache: Dict[Tuple[int,int], float] = {}
        self._it_cache: Dict[Tuple[int,int], float] = {}
        self._td_cache: Dict[Tuple[int,int], float] = {}

    def _ensure_round_cache(self):
        if getattr(self, "_rc_round", None) == self.sim.round:
            return
        S = self.sim.alive_nodes()
        # 邻居缓存
        nid2node = {n.nid: n for n in S}
        self._neigh = {}
        for i in S:
            self._neigh[i.nid] = [j for j in S if j.nid != i.nid and dist(i.pos(), j.pos()) <= COMM_RANGE]
        self._commons = {}
        # DET 缓存
        self._det_cache = {n.nid: (1.0 if n.energy >= self.Eth else 0.0) for n in S}
        # 清空其他
        self._dt_cache.clear(); self._ibt_cache.clear(); self._it_cache.clear(); self._td_cache.clear()
        self._rc_round = self.sim.round

    # ---------------------- AHC：非均匀分簇 ----------------------
    def _nearby_far_split(self, alive: List[Node]) -> Tuple[List[Node], List[Node], float]:
        if not alive: return [], [], 0.0
        N = len(alive)
        nm = max(1, int(self.pn * N))
        alive_sorted = sorted(alive, key=lambda n: dist(n.pos(), self.sim.bs))
        d0 = dist(alive_sorted[min(nm-1, N-1)].pos(), self.sim.bs)
        nearby = [n for n in alive if dist(n.pos(), self.sim.bs) <= d0]
        far = [n for n in alive if n not in nearby]
        return nearby, far, d0

    def _area_bbox(self, nodes: List[Node]) -> float:
        if not nodes: return 1.0
        xs = [n.pos()[0] for n in nodes]; ys = [n.pos()[1] for n in nodes]
        return max(1.0, (max(xs)-min(xs)) * (max(ys)-min(ys)))

    def _estimate_ndc(self, far_nodes: List[Node], nch: Node) -> int:
        if not far_nodes: return 0
        M = self._area_bbox(far_nodes)
        d2 = float(np.mean([dist(n.pos(), (nch.pos() if nch else far_nodes[0].pos())) for n in far_nodes]))
        rough = int(max(self.ndc_min, round((math.sqrt(M) / max(1.0, d2)) * math.sqrt(len(far_nodes)) / 2)))
        upper = max(1, int(self.ndc_max_ratio * len(far_nodes)))
        return max(self.ndc_min, min(rough, upper))

    # ---------------------- MOCH：多目标 CH 打分 ----------------------
    def _coverage(self, node: Node, nodes: List[Node]) -> float:
        if not nodes: return 0.0
        cnt = sum(1 for m in nodes if m.nid != node.nid and dist(node.pos(), m.pos()) <= COMM_RANGE)
        return clamp(cnt / max(1, len(nodes)-1), 0.0, 1.0)

    def _comm_cost_term(self, node: Node, nodes: List[Node]) -> float:
        neigh = [dist(node.pos(), m.pos()) for m in nodes if m.nid != node.nid and dist(node.pos(), m.pos()) <= COMM_RANGE]
        if not neigh: return 0.0
        avg = float(np.mean(neigh))
        return clamp(1.0 - avg / (COMM_RANGE + 1e-9), 0.0, 1.0)

    def _proximity_term(self, node: Node, nodes: List[Node]) -> float:
        if not nodes: return 0.0
        ds = [dist(node.pos(), m.pos()) for m in nodes if m.nid != node.nid]
        if not ds: return 0.0
        avg = float(np.mean(ds))
        dmax = max(ds)
        return clamp(1.0 - avg / (dmax + 1e-9), 0.0, 1.0)

    def _mof(self, node: Node, nodes: List[Node]) -> float:
        if not nodes: return 0.0
        e_norm = node.energy / (max(m.energy for m in nodes) + 1e-9)
        cov  = self._coverage(node, nodes)
        cost = self._comm_cost_term(node, nodes)
        prox = self._proximity_term(node, nodes)
        return (self.w_cov*cov + self.w_e*e_norm + self.w_cost*cost + self.w_prox*prox)

    # ---------------------- TAR：无副作用的评估函数 ----------------------
    def _ensure_edge_state(self, i: Node, j: Node):
        if not hasattr(i, "tr_pos"): i.tr_pos = {}
        if not hasattr(i, "tr_neg"): i.tr_neg = {}
        if not hasattr(i, "tr_t"):   i.tr_t   = {}
        if j.nid not in i.tr_pos: i.tr_pos[j.nid] = 0.0
        if j.nid not in i.tr_neg: i.tr_neg[j.nid] = 0.0
        if j.nid not in i.tr_t:   i.tr_t[j.nid]   = self.sim.round

    # —— 只在 apply_watchdog 中调用，进行时衰与写状态
    def _decay_counts_inplace(self, i: Node, j: Node):
        self._ensure_edge_state(i, j)
        dt = max(0, self.sim.round - i.tr_t[j.nid])
        i.tr_pos[j.nid] *= math.exp(-self.rho_pos * dt)
        i.tr_neg[j.nid] *= math.exp(-self.rho_neg * dt)
        i.tr_t[j.nid] = self.sim.round

    # —— 评估版：不改状态
    def _dbt_eval(self, i: Node, j: Node, id_flag: Optional[bool]=None) -> float:
        self._ensure_edge_state(i, j)
        pos = i.tr_pos[j.nid]; neg = i.tr_neg[j.nid]
        id_pos = 1.0 if id_flag is True else 0.0
        id_neg = 1.0 if id_flag is False else 0.0
        num = pos + id_pos
        den = pos + neg + id_pos + id_neg + 1e-12
        return clamp(num / den, 0.0, 1.0)

    def _det_eval(self, j: Node) -> float:
        self._ensure_round_cache()
        return self._det_cache.get(j.nid, 1.0 if j.energy >= self.Eth else 0.0)

    def _dt_eval(self, i: Node, j: Node, id_flag: Optional[bool]) -> float:
        key = (i.nid, j.nid)
        if key in self._dt_cache: return self._dt_cache[key]
        val = clamp(self.wB * self._dbt_eval(i, j, id_flag) + self.wE * self._det_eval(j), 0.0, 1.0)
        self._dt_cache[key] = val
        return val

    def _neighbors_cached(self, x: Node) -> List[Node]:
        self._ensure_round_cache()
        return self._neigh.get(x.nid, [])

    def _commons_cached(self, i: Node, j: Node) -> List[Node]:
        self._ensure_round_cache()
        key = (i.nid, j.nid)
        if key in self._commons: return self._commons[key]
        Ni = self._neigh.get(i.nid, [])
        Nj_set = {n.nid for n in self._neigh.get(j.nid, [])}
        commons = [n for n in Ni if n.nid in Nj_set]
        self._commons[key] = commons
        return commons

    def _ibt_eval(self, i: Node, j: Node, commons: List[Node]) -> float:
        key = (i.nid, j.nid)
        if key in self._ibt_cache: return self._ibt_cache[key]
        if not commons:
            self._ibt_cache[key] = 0.0
            return 0.0
        vals = [self._dbt_eval(i, k) * self._dbt_eval(k, j) for k in commons]
        val = float(np.mean(vals)) if vals else 0.0
        self._ibt_cache[key] = val
        return val

    def _cd_value(self, i: Node, j: Node, ibt: float, dt_ij: float, commons: List[Node]) -> float:
        sum_dtik = sum(self._dt_eval(i, k, None) for k in commons) if commons else 0.0
        return (ibt + dt_ij) / (sum_dtik + 1.0)

    def _it_eval(self, i: Node, j: Node) -> float:
        key = (i.nid, j.nid)
        if key in self._it_cache: return self._it_cache[key]
        commons = self._commons_cached(i, j)
        ibt0 = self._ibt_eval(i, j, commons)
        dt_ij = self._dt_eval(i, j, None)
        cd = self._cd_value(i, j, ibt0, dt_ij, commons)
        trusted_ks = [k for k in commons if abs(self._dt_eval(k, j, None) - cd) <= self.delta]
        ibt = self._ibt_eval(i, j, trusted_ks) if trusted_ks else 0.0
        val = clamp(self.uB * ibt + self.uE * self._det_eval(j), 0.0, 1.0)
        self._it_cache[key] = val
        return val

    def _td_eval(self, i: Node, j: Node, id_flag: Optional[bool]) -> float:
        key = (i.nid, j.nid)
        if key in self._td_cache: return self._td_cache[key]
        dt = self._dt_eval(i, j, id_flag)
        it = self._it_eval(i, j)
        val = clamp(self.v1 * dt + self.v2 * it, 0.0, 1.0)
        self._td_cache[key] = val
        return val

    # ---------------------- 五个钩子 ----------------------
    def select_cluster_heads(self):
        self._reset_round_cache()  # 新一轮，先清缓存
        sim = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        cand = [n for n in alive if (sim.round - n.last_ch_round) >= CH_COOLDOWN]

        nearby, far, d0 = self._nearby_far_split(cand)
        sim.clusters.clear()

        if nearby:
            nch = max(nearby, key=lambda n: self._mof(n, nearby))
            nch.is_ch = True; nch.last_ch_round = sim.round
            sim.clusters[nch.nid] = Cluster(nch.nid)
        else:
            nch = None

        if far:
            ndc = self._estimate_ndc(far, nch if nch else random.choice(far))
            ranked = sorted(far, key=lambda n: self._mof(n, far), reverse=True)
            picked: List[Node] = []
            for n in ranked:
                if len(picked) >= ndc: break
                if all(dist(n.pos(), p.pos()) >= (0.5 * COMM_RANGE) for p in picked):
                    picked.append(n)
            if not picked and ranked:
                picked = ranked[:1]
            for ch in picked:
                ch.is_ch = True; ch.last_ch_round = sim.round
                sim.clusters[ch.nid] = Cluster(ch.nid)

        if len(sim.clusters) == 0 and cand:
            pick = random.choice(cand)
            pick.is_ch = True; pick.last_ch_round = sim.round
            sim.clusters[pick.nid] = Cluster(pick.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        return False

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        self._ensure_round_cache()
        sim = self.sim
        d_bs = dist(ch.pos(), sim.bs)
        direct_cost = e_tx(DATA_PACKET_BITS, d_bs)

        cands = [x for x in ch_nodes
                 if x.alive and x.nid != ch.nid and not x.blacklisted
                 and dist(ch.pos(), x.pos()) <= CH_NEIGHBOR_RANGE]
        if not cands:
            ch.last_selected_relay = None
            return None, {}

        scored = []
        for nb in cands:
            td = self._td_eval(ch, nb, id_flag=None)
            d1 = dist(ch.pos(), nb.pos())
            d2 = dist(nb.pos(), sim.bs)
            twohop = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)
            scored.append((td, nb, d1, d2, twohop))
        scored.sort(key=lambda t: (t[0], -t[4]))  # 先信任，再能耗

        best_td, relay, d1, d2, cost2 = scored[0]
        ch.last_selected_relay = relay
        return relay, {'td': best_td, 'cost2': cost2, 'cost1': direct_cost, 'd1': d1, 'd2': d2}

    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        # 只在这里做“时衰 + 写状态”，评估阶段不再动状态
        j = getattr(ch, "last_selected_relay", None)
        if j is None:
            return
        self._decay_counts_inplace(ch, j)
        if ok and timely:
            ch.tr_pos[j.nid] += 1.0
        else:
            ch.tr_neg[j.nid] += 1.0
            ch.consecutive_strikes += 1

        if ok and timely:
            ch.observed_success += 1.0
            ch.consecutive_strikes = max(0, ch.consecutive_strikes - 1)
        else:
            ch.observed_fail += 1.0

    def finalize_trust_blacklist(self):
        self._ensure_round_cache()
        S = self.sim.alive_nodes()
        # 节点级遗忘
        for n in S:
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail
            if n.consecutive_strikes >= self.strike_threshold:
                n.blacklisted = True

        # 邻域 TD 中位数（用缓存的 _td_eval）
        for j in S:
            Ni = self._neighbors_cached(j)
            if not Ni:
                continue
            vals = [self._td_eval(i, j, None) for i in Ni]
            med = float(np.median(vals)) if vals else 1.0
            if med < self.td_blacklist:
                j.blacklisted = True
