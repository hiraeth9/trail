
# -*- coding: utf-8 -*-
from typing import List
import random, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\
                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE

class TFSM_DPC(AlgorithmBase):
    @property
    def name(self): return "TFSM-DPC-2024"
    @property
    def trust_warn(self): return 0.60
    @property
    def trust_blacklist(self): return 0.26
    @property
    def forget(self): return 0.995
    @property
    def strike_threshold(self): return 3

    def _density_peaks(self, nodes:List[Node]):
        coords=np.array([(n.x,n.y) for n in nodes])
        dc=COMM_RANGE/2.0
        dists=np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))
        rho=np.sum(np.exp(-(dists/dc)**2), axis=1) - 1.0
        delta=np.zeros(len(nodes))
        for i in range(len(nodes)):
            higher=np.where(rho>rho[i])[0]
            if len(higher)==0: delta[i]=np.max(dists[i,:])
            else: delta[i]=np.min(dists[i,higher])
        gamma=rho*delta
        return rho, delta, gamma

    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        rho, delta, gamma = self._density_peaks(alive)
        energies=np.array([n.energy for n in alive]); e_min,e_max=np.min(energies), np.max(energies)
        e_norm=(energies - e_min)/(e_max-e_min + 1e-9)
        t=np.array([n.trust() for n in alive])
        gamma_ = 0.6*gamma + 0.2*e_norm + 0.2*t
        K=max(1, int(BASE_P_CH*len(alive)))
        idxs=np.argsort(-gamma_)[:K]
        for idx in idxs:
            n=alive[idx]
            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue
            n.is_ch=True; n.last_ch_round=sim.round
            sim.clusters[n.nid]=Cluster(n.nid)
        if len(sim.clusters)<K:
            order=np.argsort(-rho)
            for idx in order:
                n=alive[idx]
                if n.nid in sim.clusters: continue
                n.is_ch=True; n.last_ch_round=sim.round; sim.clusters[n.nid]=Cluster(n.nid)
                if len(sim.clusters)>=K: break

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        return (self.trust_blacklist <= ch.trust() < self.trust_warn) and (ch.queue_level>0)

    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):
        sim=self.sim
        d_bs=dist(ch.pos(), sim.bs)
        best=None; best_score=1e18; pair=(None,None)
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue
            d1=dist(ch.pos(), other.pos())
            if d1>CH_NEIGHBOR_RANGE: continue
            d2=dist(other.pos(), sim.bs)
            score=0.30*(d1+d2)+0.35*(other.queue_level+1)+0.35*(1.0-other.trust())
            if score<best_score: best_score=score; best=other; pair=(d1,d2)
        if best is not None and (pair[0]+pair[1])<0.9*d_bs:
            return best, {'d1':pair[0],'d2':pair[1]}
        return None, {}

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        if ok and timely:
            if random.random()<0.03: ch.observed_fail+=0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)
        else:
            if random.random()<0.65: ch.observed_fail+=0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s*self.forget + n.observed_success
            n.trust_f = n.trust_f*self.forget + n.observed_fail
            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:
                n.blacklisted=True
            if n.suspicion>=0.65: n.blacklisted=True
