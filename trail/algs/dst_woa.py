
# -*- coding: utf-8 -*-
from typing import List, Tuple
import random, math, numpy as np
from core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\
                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS, e_tx, e_rx

def ds_fuse(t:float, prior:float=0.5)->float:
    m1_g, m1_b = t, 1.0-t
    m2_g, m2_b = prior, 1.0-prior
    K = m1_g*m2_b + m1_b*m2_g
    if K >= 0.999999: return t
    return (m1_g*m2_g) / (1.0 - K)

class DST_WOA(AlgorithmBase):
    @property
    def name(self): return "DST-WOA-2024"
    @property
    def trust_warn(self): return 0.60
    @property
    def trust_blacklist(self): return 0.26
    @property
    def forget(self): return 0.995
    @property
    def strike_threshold(self): return 3

    def _evaluate_binary_chset(self, bitvec:np.ndarray)->float:
        sim=self.sim
        ch_idx=np.where(bitvec>0.5)[0]
        if len(ch_idx)==0: return 1e12
        ch_nodes=[sim.nodes[i] for i in ch_idx if (sim.nodes[i].alive and not sim.nodes[i].blacklisted)]
        if not ch_nodes: return 1e12
        cost_intra=0.0; penalty=0.0
        for n in sim.nodes:
            if n in ch_nodes: continue
            dmin=1e9
            for ch in ch_nodes:
                d=dist(n.pos(), ch.pos()); dmin=min(dmin,d)
            if dmin>COMM_RANGE: penalty+=1e6; continue
            cost_intra += e_tx(DATA_PACKET_BITS, dmin) + e_rx(DATA_PACKET_BITS)*0.5
        cost_up=sum(e_tx(DATA_PACKET_BITS, dist(ch.pos(), sim.bs)) for ch in ch_nodes)
        trust_penalty=sum((1.0-ds_fuse(ch.trust(),0.5))*1e-4 for ch in ch_nodes)
        return cost_intra + cost_up + penalty + trust_penalty

    def _binary_woa(self, dim:int, K:int):
        whales=12; iters=15; rnd=random.Random(42)
        pop=np.clip(np.random.rand(whales, dim),0,1)
        for i in range(whales):
            idx=np.argsort(-pop[i])[:K]; vec=np.zeros(dim); vec[idx]=1.0; pop[i]=vec
        fitness=np.array([self._evaluate_binary_chset(pop[i]) for i in range(whales)])
        bi=int(np.argmin(fitness)); best=pop[bi].copy(); best_fit=float(fitness[bi])
        for t in range(iters):
            a=2 - 2*t/iters
            for i in range(whales):
                r1, r2 = rnd.random(), rnd.random()
                A = 2*a*r1 - a; C = 2*r2
                p = rnd.random()
                if p<0.5:
                    if abs(A)>=1:
                        j=rnd.randrange(whales); D=np.abs(C*pop[j]-pop[i]); new=pop[j]-A*D
                    else:
                        D=np.abs(C*best - pop[i]); new=best - A*D
                else:
                    b=1; l=(rnd.random()*2-1); D=np.abs(best - pop[i]); new = D*np.e**(b*l)*np.cos(2*np.pi*l)+best
                s=1/(1+np.exp(-new)); idx=np.argsort(-s)[:K]; vec=np.zeros(dim); vec[idx]=1.0; pop[i]=vec
            fitness=np.array([self._evaluate_binary_chset(pop[i]) for i in range(whales)])
            j=int(np.argmin(fitness))
            if fitness[j] < best_fit: best_fit=float(fitness[j]); best=pop[j].copy()
        return best

    def select_cluster_heads(self):
        sim=self.sim
        alive=[n for n in sim.alive_nodes() if not n.blacklisted]
        if not alive: return
        dim=len(sim.nodes); K=max(1, int(BASE_P_CH*len(alive)))
        bitvec=self._binary_woa(dim, K)
        for i,v in enumerate(bitvec):
            if v>0.5 and (sim.nodes[i] in alive) and ((sim.round - sim.nodes[i].last_ch_round) >= CH_COOLDOWN):
                sim.nodes[i].is_ch=True; sim.nodes[i].last_ch_round=sim.round
                sim.clusters[sim.nodes[i].nid]=Cluster(sim.nodes[i].nid)

    def allow_member_redundancy(self, member:Node, ch:Node)->bool:
        return False

    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):
        sim=self.sim
        d_bs=dist(ch.pos(), sim.bs)
        best=None; best_score=1e18; pair=(None,None)
        for other in ch_nodes:
            if other.nid==ch.nid or (not other.alive): continue
            d1=dist(ch.pos(), other.pos()); 
            if d1>CH_NEIGHBOR_RANGE: continue
            d2=dist(other.pos(), sim.bs)
            t_fused = ds_fuse(other.trust(), 0.5)
            score=0.30*(d1+d2)+0.45*(1.0-t_fused)+0.15*(other.queue_level+1)+0.10*(1.0-(other.energy/sim.nodes[0].energy))
            if score<best_score: best_score=score; best=other; pair=(d1,d2)
        if best is not None and (pair[0]+pair[1])<0.9*d_bs:
            return best, {'d1':pair[0],'d2':pair[1]}
        return None, {}

    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):
        if ok and timely:
            if random.random()<0.03: ch.observed_fail+=0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)
        else:
            if random.random()<0.70: ch.observed_fail+=0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)

    def finalize_trust_blacklist(self):
        sim=self.sim
        for n in sim.alive_nodes():
            n.trust_s = n.trust_s*self.forget + n.observed_success
            n.trust_f = n.trust_f*self.forget + n.observed_fail
            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:
                n.blacklisted=True
            if n.suspicion>=0.65: n.blacklisted=True
