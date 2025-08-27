
# -*- coding: utf-8 -*-
from typing import List
import random, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, e_tx, e_rx, clamp,\
                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS

def tri(x, a, b, c):
    if x<=a or x>=c: return 0.0
    if x==b: return 1.0
    return (x-a)/(b-a) if x<a or x<b else (c-x)/(c-b)

class SHEER(AlgorithmBase):
    @property
    def name(self): return "SHEER-2025"
    @property
    def trust_warn(self): return 0.65
    @property
    def trust_blacklist(self): return 0.25
    @property
    def forget(self): return 0.98
    @property
    def strike_threshold(self): return 3

    def _fis_score(self, e_norm, t, b_norm):
        eL,eM,eH = tri(e_norm,0.0,0.0,0.5), tri(e_norm,0.25,0.5,0.75), tri(e_norm,0.5,1.0,1.0)
        tL,tM,tH = tri(t,0.0,0.0,0.5), tri(t,0.25,0.5,0.75), tri(t,0.5,1.0,1.0)
        bN,bM,bF = tri(b_norm,0.5,1.0,1.0), tri(b_norm,0.25,0.5,0.75), tri(b_norm,0.0,0.0,0.5)
        rules = []
        rules += [max(eH,tH,bN)*0.95, max(eH,tM,bN)*0.9, max(eM,tH,bN)*0.9]
        rules += [max(eM,tM,bM)*0.7, max(eH,tM,bM)*0.8, max(eM,tH,bM)*0.8]
        rules += [max(eL,tL,bF)*0.2, max(eL,tM,bF)*0.4, max(eM,tL,bF)*0.4]
        num = sum(rules); den = len(rules) if len(rules)>0 else 1
        return num/den

    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)
        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)
        for n in alive:
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            e_norm=(n.energy - e_min)/(e_max-e_min + 1e-9)
            t=n.trust()
            b=1.0 - (dist(n.pos(), sim.bs)-d_min)/(d_max-d_min + 1e-9)
            score = self._fis_score(e_norm, t, b)
            p=clamp(BASE_P_CH*(0.5 + 0.8*score), 0.0, 1.0)
            if random.random()<p:
                n.is_ch=True; n.last_ch_round=sim.round
                sim.clusters[n.nid]=Cluster(ch_id=n.nid)
        if len(sim.clusters)<max(1, int(0.03*len(alive))):
            cand=sorted([n for n in alive if not n.is_ch],
                        key=lambda x: self._fis_score((x.energy-e_min)/(e_max-e_min+1e-9), x.trust(),
                                                      1.0-(dist(x.pos(), sim.bs)-d_min)/(d_max-d_min+1e-9)),
                        reverse=True)
            for c in cand[:max(1,int(0.03*len(alive)))-len(sim.clusters)]:
                c.is_ch=True; c.last_ch_round=sim.round; sim.clusters[c.nid]=Cluster(c.nid)

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        return False

    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):
        sim=self.sim
        d_bs=dist(ch.pos(), sim.bs)
        best=None; best_cost=1e9; best_pair=(None,None)
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue
            d1=dist(ch.pos(), other.pos())
            if d1>CH_NEIGHBOR_RANGE: continue
            d2=dist(other.pos(), sim.bs)
            cost=0.35*(d1+d2)+0.35*(1.0-other.trust())+0.30*(other.queue_level+1)
            if cost<best_cost:
                best_cost=cost; best=other; best_pair=(d1,d2)
        use = (best is not None) and ((best_pair[0]+best_pair[1]) < 0.90*d_bs or ch.trust()<self.trust_warn)
        return (best if use else None), ({'d1':best_pair[0],'d2':best_pair[1]} if use else {} )

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        pass

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s*self.forget + n.observed_success
            n.trust_f = n.trust_f*self.forget + n.observed_fail
            if n.trust()<self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:
                n.blacklisted=True
