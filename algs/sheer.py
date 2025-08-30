# -*- coding: utf-8 -*-
from typing import List, Tuple
import random, numpy as np
from core.wsn_core import (
    AlgorithmBase, Simulation, Node, Cluster, dist, e_tx, e_rx, clamp,
    BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS
)

class SHEER(AlgorithmBase):
    """
    SHEER (paper-aligned):
      - CH fitness:  CS = α·Eres/Emax + β·(1 - Dprox/Dmax) + γ·(1 - Dsink/DmaxSink) + δ·Tnode
      - CH→CH relay fitness: λ1·Eres/Emax + λ2·(1 - D_CH-CH/Dmax) + λ3·T_CH
      - Self-healing: energy/time threshold based (Eq.7 semantics)
    """

    # ---- 参数（可按实验需要微调） ----
    # 论文式(6)权重
    alpha_e = 0.35    # 能量
    beta_p  = 0.25    # 邻近性/连通性
    gamma_s = 0.20    # 到汇聚节点距离（更近更优）
    delta_t = 0.20    # 信任

    # 论文式(8)权重（CH→CH）
    lam_e = 0.40
    lam_d = 0.25
    lam_t = 0.35

    # CH 资格的信任阈值（论文 0.7）
    ch_trust_threshold = 0.70

    # 自愈阈值（式(7)语义）：能量比例阈值与“非及时达”判据
    energy_fail_ratio = 0.10   # E_res <= 10%·E_init 视为能量击穿
    # 时延阈值由主程序传入的 timely=False 体现

    # 两跳能耗容忍系数：两跳 <= η · 直达 才启用中继
    eta_twohop = 1.08

    @property
    def name(self): return "SHEER-2025"

    @property
    def trust_warn(self): return 0.80   # 用于内部保守判据
    @property
    def trust_blacklist(self): return 0.35  # 维持成员选择下限，避免过度拉黑
    @property
    def forget(self): return 0.98
    @property
    def strike_threshold(self): return 3

    # ---------- 内部工具函数 ----------

    def _neighbors_within(self, node: Node, alive: List[Node], radius: float):
        out = []
        p0 = node.pos()
        for other in alive:
            if other.nid == node.nid or (not other.alive): continue
            d = dist(p0, other.pos())
            if d <= radius: out.append((d, other))
        return out

    def _compute_proximity_term(self, node: Node, alive: List[Node]) -> float:
        """
        D_prox: 节点到“通信半径内邻居”的平均距离；无邻居则按最差处理 (=D_max)
        返回 (1 - D_prox/D_max) ∈ [0,1]
        """
        neigh = self._neighbors_within(node, alive, COMM_RANGE)
        if not neigh: return 0.0
        avg_d = float(np.mean([d for d, _ in neigh]))
        return clamp(1.0 - (avg_d / (COMM_RANGE + 1e-9)), 0.0, 1.0)

    def _compute_sink_term(self, node: Node, alive: List[Node], d_sink_max: float) -> float:
        """返回 (1 - D_sink/D_maxSink)"""
        d = dist(node.pos(), self.sim.bs)
        return clamp(1.0 - d / (d_sink_max + 1e-9), 0.0, 1.0)

    def _compute_trust_node(self, node: Node) -> float:
        """
        论文式(4)(5)：T = ω1·T_direct(PDR) + ω2·T_indirect(C_hist)；此处利用框架已有
        Beta-估计 trust() 作为主项，并轻度融合 direct/relay 的历史值（若存在）。
        """
        base = node.trust()             # Beta先验 + 观测（主程序在 finalize 中更新）
        # direct/relay 历史：框架中其它算法会更新 dir_val / rly_val，范围[0,1]；SHEER 也可受益
        if hasattr(node, "dir_val") and hasattr(node, "rly_val"):
            hist = 0.5 * float(node.dir_val) + 0.5 * float(node.rly_val)
            return clamp(0.6 * base + 0.4 * hist, 0.0, 1.0)
        return base

    def _fitness_ch(self, node: Node, alive: List[Node], e_max: float, d_sink_max: float) -> float:
        e_term = clamp(node.energy / (e_max + 1e-9), 0.0, 1.0)
        p_term = self._compute_proximity_term(node, alive)
        s_term = self._compute_sink_term(node, alive, d_sink_max)
        t_term = self._compute_trust_node(node)
        return (self.alpha_e * e_term +
                self.beta_p  * p_term +
                self.gamma_s * s_term +
                self.delta_t * t_term)

    # ---------- 必要接口实现 ----------

    def select_cluster_heads(self):
        sim = self.sim
        alive = [n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return

        e_max = max(n.energy for n in alive)
        d_sink_max = max(dist(n.pos(), sim.bs) for n in alive)

        # 1) 概率抽样（由 CS 归一后映射），保持与主流程兼容
        scored = []
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            # 论文：只有 trust ≥ 0.7 才具备 CH 资格
            if self._compute_trust_node(n) < self.ch_trust_threshold:  # Eq.(5)阈值
                continue
            cs = self._fitness_ch(n, alive, e_max, d_sink_max)         # Eq.(6)
            scored.append((cs, n))
            # 将 CS 映射为抽样概率（与 BASE_P_CH 协同），并抽样
            p = clamp(BASE_P_CH * (0.60 + 0.80 * cs), 0.0, 1.0)
            if random.random() < p:
                n.is_ch = True; n.last_ch_round = sim.round
                sim.clusters[n.nid] = Cluster(n.nid)

        # 2) 保底：按 CS 从高到低补齐到期望下限（例如 ≥8% 活跃节点）
        need_total = max(1, int(0.08 * len(alive)))
        if len(sim.clusters) < need_total:
            if not scored:
                # 没有任何候选，放宽一次资格用于极端场景
                for n in alive:
                    cs = self._fitness_ch(n, alive, e_max, d_sink_max)
                    scored.append((cs, n))
            scored.sort(key=lambda x: x[0], reverse=True)
            for _, pick in scored[: (need_total - len(sim.clusters))]:
                if pick.nid not in sim.clusters:
                    pick.is_ch = True; pick.last_ch_round = sim.round
                    sim.clusters[pick.nid] = Cluster(pick.nid)

    def allow_member_redundancy(self, member: Node, ch: Node) -> bool:
        # 论文未主张成员侧的多路径冗余，这里保持关闭以贴合能耗模型
        return False

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        """
        论文式(8)：在邻近 CH 集合内，按 (能量/距离/信任) 的通信适应度选择下一跳 CH。
        同时用能耗门槛保护：两跳总能耗不应明显劣于直达。
        """
        sim = self.sim
        # 直达代价
        d_bs = dist(ch.pos(), sim.bs)
        cost_direct = e_tx(DATA_PACKET_BITS, d_bs)

        # 候选邻居
        cands = []
        pool = [x for x in ch_nodes if x.alive and x.nid != ch.nid]
        if not pool:
            return None, {}

        e_max = max(x.energy for x in pool) if pool else ch.energy
        for other in pool:
            d1 = dist(ch.pos(), other.pos())
            if d1 > CH_NEIGHBOR_RANGE:       # 邻接范围内才可作为下一跳
                continue
            # 论文式(8)的三项
            e_term = clamp(other.energy / (e_max + 1e-9), 0.0, 1.0)
            d_term = clamp(1.0 - d1 / (CH_NEIGHBOR_RANGE + 1e-9), 0.0, 1.0)
            t_term = self._compute_trust_node(other)

            fit = self.lam_e * e_term + self.lam_d * d_term + self.lam_t * t_term  # Eq.(8)

            # 两跳真实能耗（与主程序能量模型一致）
            d2 = dist(other.pos(), sim.bs)
            cost_twohop = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)

            cands.append((fit, other, d1, d2, cost_twohop))

        if not cands:
            return None, {}

        # 适应度最大者优先；同时用能耗门槛筛选
        cands.sort(key=lambda x: x[0], reverse=True)
        best_fit, relay, d1, d2, cost2 = cands[0]

        use_relay = (relay is not None) and (self._compute_trust_node(relay) >= self.ch_trust_threshold) \
                    and (cost2 <= self.eta_twohop * cost_direct)

        return (relay if use_relay else None), (
            {'d1': d1, 'd2': d2, 'cost2': cost2, 'cost1': cost_direct, 'fit': best_fit} if use_relay else {}
        )

    def apply_watchdog(self, ch: Node, ok: bool, timely: bool, ch_nodes: List[Node]):
        """
        自愈监测（式(7)）：能量低或非及时达 → 视为一次击穿，叠加可疑/失败计数；
        正常及时达则衰减可疑。黑名单与信任更新在 finalize 中统一处理。
        """
        # 能量判据：相对初始能量比例（主程序 INIT_ENERGY=0.5，此处用相对比例更稳健）
        # 由于 INIT_ENERGY 未直接暴露，改用“全网存活节点的能量中位数”近似动态阈值
        alive = self.sim.alive_nodes()
        med_e = float(np.median([n.energy for n in alive])) if alive else 0.0
        energy_breach = (ch.energy <= max(1e-6, self.energy_fail_ratio * med_e))

        time_breach = (not timely)  # 主程序把延期编为 timely=False

        if (not ok) or energy_breach or time_breach:
            ch.observed_fail += 0.5
            ch.suspicion = min(1.0, ch.suspicion + (0.20 if energy_breach else 0.12))
            ch.consecutive_strikes += 1
        else:
            ch.observed_success += 0.6
            ch.suspicion = max(0.0, ch.suspicion * 0.85)
            ch.consecutive_strikes = max(0, ch.consecutive_strikes - 1)

    def finalize_trust_blacklist(self):
        """
        信任更新 + 黑名单：指数遗忘 + 观测叠加 + 可疑轻惩；
        拉黑条件：低信任并伴随击穿，或击穿累计超阈值（与主程序风格一致）。
        """
        for n in self.sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail + 0.15 * n.suspicion
            low_trust = (n.trust() < self.trust_blacklist)
            if (low_trust and n.consecutive_strikes >= 1) or (n.consecutive_strikes >= self.strike_threshold):
                n.blacklisted = True
