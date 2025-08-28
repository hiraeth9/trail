
# -*- coding: utf-8 -*-
from typing import List
import random, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp, \
    BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, \
    e_tx, e_rx, DATA_PACKET_BITS

class TRAIL(AlgorithmBase):
    @property
    def name(self): return "TRAIL (ours)"
    @property
    def trust_warn(self): return 0.55
    @property
    def trust_blacklist(self): return 0.35

    @property
    def forget(self):
        return 0.97

    @property
    def strike_threshold(self):
        return 3

    def __init__(self, sim: 'Simulation'):
        super().__init__(sim)
        # —— 探索/冗余/能耗-可靠性策略参数 ——
        self.rt_ratio = 0.24  # 仅在CH“临界可信”时才考虑冗余，基础概率更低
        self.epsilon = 0.22  # 初始探索率↑，更快发现好中继
        self.eps_decay = 0.996  # 探索缓慢衰减
        self.min_epsilon = 0.04
        self.alpha = 1.02  # 允许两跳能耗略高于直达，换可靠性/吞吐
        self.queue_pen = 0.03  # 中继队列惩罚系数
        self.trust_pen = 0.045  # 中继低信任惩罚
        self.rel_w = 0.35  # 可靠性记忆(直达BS成功率)的权重
        self.mode_by_ch = {}  # 记录上一轮CH采用 direct/relay

    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)
        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)
        q_levels=[n.last_queue_level for n in alive]; q_min, q_max=min(q_levels), max(q_levels)
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            e = (n.energy - e_min) / (e_max - e_min + 1e-9)
            t = n.trust()
            q = 1.0 - (n.last_queue_level - q_min) / (q_max - q_min + 1e-9)
            b = 1.0 - (dist(n.pos(), sim.bs) - d_min) / (d_max - d_min + 1e-9)
            s_pen = 1.0 - max(0.0, min(1.0, 1.0 - n.suspicion))
            # 能量优先，怀疑惩罚减半，轻度加权“更近BS”的中心性
            score = 0.44 * e + 0.26 * (t - 0.10 * s_pen) + 0.15 * q + 0.15 * b
            p = clamp(BASE_P_CH * (0.62 + score), 0.0, 1.0)
            if random.random()<p:
                n.is_ch=True; n.last_ch_round=sim.round
                sim.clusters[n.nid]=Cluster(n.nid)
        # 若CH数低于存活节点的 ~3%，再补选若干个分数最高的
        if len(sim.clusters) < max(1, int(0.03 * len(alive))):
            scored = []
            for nn in [x for x in alive if not x.is_ch]:
                ee = (nn.energy - e_min) / (e_max - e_min + 1e-9)
                tt = nn.trust()
                qq = 1.0 - (nn.last_queue_level - q_min) / (q_max - q_min + 1e-9)
                bb = 1.0 - (dist(nn.pos(), sim.bs) - d_min) / (d_max - d_min + 1e-9)
                ss = 1.0 - max(0.0, min(1.0, 1.0 - nn.suspicion))
                sc = 0.44 * ee + 0.26 * (tt - 0.10 * ss) + 0.15 * qq + 0.15 * bb
                scored.append((sc, nn))
            scored.sort(key=lambda x: x[0], reverse=True)
            need = max(1, int(0.03 * len(alive))) - len(sim.clusters)
            for _, pick in scored[:need]:
                pick.is_ch = True;
                pick.last_ch_round = sim.round;
                sim.clusters[pick.nid] = Cluster(pick.nid)

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        alive = self.sim.alive_nodes()
        if not (self.trust_blacklist <= ch.trust() < self.trust_warn):
            return False
        if not alive:
            return False
        import numpy as _np
        median_e = float(_np.median([n.energy for n in alive]))
        if member.energy < median_e:  # 低电成员不做冗余，保寿命
            return False
        prob = self.rt_ratio * (1.0 - ch.trust())  # CH 越可疑，冗余概率越高
        return (random.random() < prob)

    def choose_ch_relay(self, ch: Node, ch_nodes: List[Node]):
        sim = self.sim
        d_bs = dist(ch.pos(), sim.bs)
        cost_direct = e_tx(DATA_PACKET_BITS, d_bs)
        cands = []
        for other in ch_nodes:
            if other.nid == ch.nid or (not other.alive) or other.trust() < self.trust_blacklist:
                continue
            d1 = dist(ch.pos(), other.pos())
            if d1 > CH_NEIGHBOR_RANGE:
                continue
            d2 = dist(other.pos(), sim.bs)
            two_hop_cost = e_tx(DATA_PACKET_BITS, d1) + e_rx(DATA_PACKET_BITS) + e_tx(DATA_PACKET_BITS, d2)
            # 队列/信任惩罚；可靠性记忆(该中继“直达BS”历史成功率)做奖励
            rel = other.dir_val  # [0,1] 的EWMA
            penalty = (1.0 + self.queue_pen * other.queue_level + self.trust_pen * (1.0 - other.trust()))
            reward = max(0.70, (1.0 - self.rel_w * rel))  # 最多降低 35% 的“有效成本”
            score = two_hop_cost * penalty * reward
            cands.append((score, other, d1, d2, two_hop_cost))
        if not cands:
            self.mode_by_ch[ch.nid] = 'direct'
            return None, {}
        cands.sort(key=lambda x: x[0])
        best = cands[0];
        relay = best[1];
        d1, b_d2 = best[2], best[3];
        best_cost = best[4]
        # 采用ε-贪心探索
        explore = (random.random() < self.epsilon)
        if explore:
            use_relay = (random.random() < 0.5)
        else:
            use_relay = (best_cost + 1e-12) < (self.alpha * cost_direct) and (
                        relay.queue_level <= ch.queue_level + 2) and (relay.energy > 0.08)
        # 衰减探索率
        self.epsilon = max(self.min_epsilon, self.epsilon * self.eps_decay)
        self.mode_by_ch[ch.nid] = ('relay' if use_relay else 'direct')
        return (relay if use_relay else None), (
            {'d1': d1, 'd2': b_d2, 'cost2': best_cost, 'cost1': cost_direct} if use_relay else {})

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        neigh=[]
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive): continue
            neigh.append((dist(ch.pos(), other.pos()), other))
        neigh.sort(key=lambda x: x[0]);
        watchers = [p[1] for p in neigh[:3]]  # 多一个观察者更稳
        mode = self.mode_by_ch.get(ch.nid, 'direct')
        for _w in watchers:
            if ok and timely:
                # 成功：降低可疑，增加可靠性记忆
                ch.suspicion = max(0.0, ch.suspicion - 0.04)
                if mode == 'direct':
                    ch.dir_val = 0.90 * ch.dir_val + 0.10 * 1.0;
                    ch.dir_cnt += 1
                else:
                    ch.rly_val = 0.90 * ch.rly_val + 0.10 * 1.0;
                    ch.rly_cnt += 1
            else:
                # 失败：有限度地增加可疑，并轻微衰减可靠性
                if random.random() < 0.65:
                    ch.observed_fail += 0.55
                    ch.suspicion = min(1.0, ch.suspicion + 0.18)
                ch.dir_val *= 0.95;
                ch.rly_val *= 0.95

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s * self.forget + n.observed_success
            n.trust_f = n.trust_f * self.forget + n.observed_fail + 0.20 * n.suspicion
            # 更稳健：需要“至少一次击穿”配合低信任，或击穿累计超阈值；亦或高可疑且出现过击穿
            if (n.trust() < self.trust_blacklist and n.consecutive_strikes >= 1) or \
                    (n.consecutive_strikes >= self.strike_threshold) or \
                    (n.suspicion >= 0.85 and n.consecutive_strikes >= 1):
                n.blacklisted = True

