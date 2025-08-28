
# -*- coding: utf-8 -*-
from typing import List
import random, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\
                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE

class ACTAR(AlgorithmBase):
    @property
    def name(self): return "ACTAR-2024"
    @property
    def trust_warn(self): return 0.60
    @property
    def trust_blacklist(self): return 0.26
    @property
    def forget(self): return 0.995
    @property
    def strike_threshold(self): return 4

    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)
        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)
        q_levels=[n.last_queue_level for n in alive]; q_min, q_max=min(q_levels), max(q_levels)
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            d_norm=(dist(n.pos(), sim.bs)-d_min)/(d_max-d_min+1e-9)
            ring_scale = 1.20 - 0.40*d_norm
            e=(n.energy-e_min)/(e_max-e_min+1e-9)
            t=n.trust()
            b=1.0 - d_norm
            prox=1.0 - (n.last_queue_level-q_min)/(q_max-q_min+1e-9)
            score=0.35*e + 0.30*t + 0.20*b + 0.15*prox
            p=clamp(BASE_P_CH*ring_scale*(0.5+score),0.0,1.0)
            if random.random()<p:
                n.is_ch=True; n.last_ch_round=sim.round
                sim.clusters[n.nid]=Cluster(n.nid)

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        return False

    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):
        sim=self.sim
        d_bs=dist(ch.pos(), sim.bs)
        best=None; best_sum=1e9; pair=(None,None)
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue
            d1=dist(ch.pos(), other.pos()); 
            if d1>CH_NEIGHBOR_RANGE: continue
            d2=dist(other.pos(), sim.bs)
            if d1+d2<best_sum: best_sum=d1+d2; best=other; pair=(d1,d2)
        if best is not None and best_sum < 0.9*d_bs and best.trust()>=self.trust_warn:
            return best, {'d1':pair[0],'d2':pair[1]}
        return None, {}

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        if ok and timely: 
            if random.random()<0.03: ch.observed_fail += 0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)
        else:
            if random.random()<0.60: ch.observed_fail += 0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s*self.forget + n.observed_success
            n.trust_f = n.trust_f*self.forget + n.observed_fail
            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:
                n.blacklisted = True
            if n.suspicion >= 0.65: n.blacklisted=True
