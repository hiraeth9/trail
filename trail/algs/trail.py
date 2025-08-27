
# -*- coding: utf-8 -*-
from typing import List
import random, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\
                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE

class TRAIL(AlgorithmBase):
    @property
    def name(self): return "TRAIL (ours)"
    @property
    def trust_warn(self): return 0.55
    @property
    def trust_blacklist(self): return 0.35
    @property
    def forget(self): return 0.93
    @property
    def strike_threshold(self): return 2

    def __init__(self, sim:'Simulation'):
        super().__init__(sim)
        self.rt_ratio=0.35; self.epsilon=0.15

    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)
        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)
        q_levels=[n.last_queue_level for n in alive]; q_min, q_max=min(q_levels), max(q_levels)
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            e=(n.energy-e_min)/(e_max-e_min+1e-9)
            t=n.trust()
            q=1.0 - (n.last_queue_level - q_min)/(q_max - q_min + 1e-9)
            b=1.0 - (dist(n.pos(), sim.bs)-d_min)/(d_max-d_min+1e-9)
            s_pen = 1.0 - max(0.0, min(1.0, 1.0-n.suspicion))
            score=0.35*e + 0.35*(t - 0.2*s_pen) + 0.20*q + 0.10*b
            p=clamp(BASE_P_CH*(0.5+score),0.0,1.0)
            if random.random()<p:
                n.is_ch=True; n.last_ch_round=sim.round
                sim.clusters[n.nid]=Cluster(n.nid)

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        is_rt = (random.random()<self.rt_ratio)
        return is_rt and (self.trust_blacklist <= ch.trust() < self.trust_warn)

    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):
        import math
        sim=self.sim
        d_bs=dist(ch.pos(), sim.bs); cost_direct=d_bs
        cands=[]
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue
            d1=dist(ch.pos(), other.pos())
            if d1>CH_NEIGHBOR_RANGE: continue
            d2=dist(other.pos(), sim.bs)
            score=0.35*(d1+d2)+0.30*(other.queue_level+1)+0.25*(1.0-other.trust())+0.10*(1.0-(other.energy/ (sim.nodes[0].energy if len(sim.nodes)>0 else 0.5)))
            cands.append((score, other, d1, d2))
        if not cands: return None, {}
        cands.sort(key=lambda x:x[0])
        best=cands[0]; relay=best[1]; d1,b_d2=best[2],best[3]
        explore = (random.random()<self.epsilon)
        if explore:
            use_relay = (random.random()<0.5)
        else:
            use_relay = (d1+b_d2+1e-6) < 0.95*cost_direct
        return (relay if use_relay else None), {'d1':d1,'d2':b_d2} if use_relay else {}

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        neigh=[]
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive): continue
            neigh.append((dist(ch.pos(), other.pos()), other))
        neigh.sort(key=lambda x:x[0]); watchers=[p[1] for p in neigh[:2]]
        for w in watchers:
            if ok and timely:
                if random.random()<0.05:
                    ch.observed_fail += 0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)
            else:
                if random.random()<0.75:
                    ch.observed_fail += 0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s*self.forget + n.observed_success
            n.trust_f = n.trust_f*self.forget + n.observed_fail + 0.3*n.suspicion
            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:
                n.blacklisted=True
