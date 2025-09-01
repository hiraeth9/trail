
# -*- coding: utf-8 -*-
import os, sys, argparse, importlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# FILES = {'core/wsn_core.py': '\n# -*- coding: utf-8 -*-\nimport math, random, os\nfrom dataclasses import dataclass, field\nfrom typing import List, Dict, Tuple, Optional, Any\nimport numpy as np\nimport pandas as pd\n\nSEED = 42\nAREA_W, AREA_H = 100.0, 100.0\nBS_POS = (50.0, 150.0)\nCOMM_RANGE = 30.0\nCH_NEIGHBOR_RANGE = 60.0\nINIT_ENERGY = 0.5\n\nE_ELEC = 50e-9; E_FS = 10e-12; E_MP = 0.0013e-12\nD0 = math.sqrt(E_FS / E_MP); E_DA = 5e-9\nDATA_PACKET_BITS = 4000; CTRL_PACKET_BITS = 200\n\nSIM_ROUNDS = 500\nBASE_P_CH = 0.07\nCH_COOLDOWN = int(1.0/BASE_P_CH)\n\nP_MAL_MEMBER_DROP = 0.25\nP_MAL_CH_DROP = 0.60\nP_MAL_CH_DELAY = 0.30\n\ndef dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:\n    return math.hypot(a[0]-b[0], a[1]-b[1])\n\ndef e_tx(bits: int, d: float) -> float:\n    return E_ELEC*bits + (E_FS*bits*(d**2) if d < D0 else E_MP*bits*(d**4))\n\ndef e_rx(bits: int) -> float:\n    return E_ELEC*bits\n\ndef clamp(x, lo, hi): return max(lo, min(hi, x))\n\n@dataclass\nclass Node:\n    nid: int; x: float; y: float; node_type: str\n    energy: float = INIT_ENERGY\n    trust_s: float = 0.0; trust_f: float = 0.0\n    suspicion: float = 0.0\n    consecutive_strikes: int = 0\n    blacklisted: bool = False\n    last_ch_round: int = -9999\n    alive: bool = True\n    is_ch: bool = False; cluster_id: Optional[int] = None\n    observed_success: float = 0.0; observed_fail: float = 0.0\n    queue_level: int = 0; last_queue_level: int = 0\n    dir_cnt: int = 0; dir_val: float = 0.0\n    rly_cnt: int = 0; rly_val: float = 0.0\n    def pos(self): return (self.x, self.y)\n    def trust(self): return (self.trust_s + 1.0) / (self.trust_s + self.trust_f + 2.0)\n    def reset_round_flags(self):\n        self.is_ch=False; self.cluster_id=None\n        self.observed_success=0.0; self.observed_fail=0.0\n        self.queue_level=0; self.suspicion=max(0.0, self.suspicion*0.9)\n\n@dataclass\nclass Cluster:\n    ch_id: int; members: List[int] = field(default_factory=list)\n\nclass AlgorithmBase:\n    def __init__(self, sim:\'Simulation\'): self.sim = sim\n    @property\n    def name(self)->str: return "BaseAlgo"\n    @property\n    def trust_warn(self)->float: raise NotImplementedError\n    @property\n    def trust_blacklist(self)->float: raise NotImplementedError\n    @property\n    def forget(self)->float: return 0.98\n    @property\n    def strike_threshold(self)->int: return 3\n    def suspicion_blacklist(self)->Optional[float]: return None\n    def select_cluster_heads(self): raise NotImplementedError\n    def allow_member_redundancy(self, member:Node, ch:Node)->bool: return False\n    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):\n        return None, {}\n    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]): pass\n    def finalize_trust_blacklist(self): raise NotImplementedError\n\nclass Simulation:\n    def __init__(self, algo_ctor, n_nodes:int=100, n_malicious:int=30, seed:int=SEED):\n        self.n_nodes=n_nodes; self.n_malicious=n_malicious\n        self.bs=BS_POS; self.round=0; self.seed=seed\n        self.nodes: List[Node]=[]; self.clusters: Dict[int, Cluster]={}\n        self.history=[]; self.rand=random.Random(seed)\n        self._init_nodes()\n        self.total_drop=0; self.total_delivered=0; self.total_timely_delivered=0\n        self.malicious_delay_times=0; self.malicious_drop_packets=0\n        self.member_energy_total=0.0; self.member_energy_effective=0.0\n        self.total_energy_used=0.0; self.total_control_bits=0; self.total_hop_count=0\n        self.sum_cluster_size=0; self.count_cluster_rounds=0; self.sum_ch_count=0\n        self.FND=None; self.HND=None; self.LND=None\n        self.algo: AlgorithmBase = algo_ctor(self)\n\n    def _init_nodes(self):\n        coords = np.column_stack([np.random.rand(self.n_nodes)*AREA_W,\n                                  np.random.rand(self.n_nodes)*AREA_H])\n        mal = set(self.rand.sample(range(self.n_nodes), self.n_malicious))\n        for i in range(self.n_nodes):\n            t = "malicious" if i in mal else "normal"\n            self.nodes.append(Node(i, float(coords[i,0]), float(coords[i,1]), t))\n\n    def alive_nodes(self): return [n for n in self.nodes if n.alive]\n\n    def elect_cluster_heads(self):\n        self.clusters={}\n        for n in self.nodes: n.reset_round_flags()\n        self.algo.select_cluster_heads()\n        if len(self.clusters)==0:\n            alive=[n for n in self.alive_nodes() if not n.blacklisted] or self.alive_nodes()\n            if alive:\n                ch=max(alive, key=lambda x:x.energy)\n                ch.is_ch=True; ch.last_ch_round=self.round; self.clusters[ch.nid]=Cluster(ch.nid)\n\n    def assign_members(self):\n        alive=self.alive_nodes(); chs=[n for n in alive if n.is_ch]\n        if not chs: return\n        for n in alive:\n            if n.is_ch: continue\n            cands=[]\n            for ch in chs:\n                if ch.trust()<self.algo.trust_blacklist: continue\n                d=dist(n.pos(), ch.pos())\n                if d<=COMM_RANGE: cands.append((d,ch))\n            if not cands: n.cluster_id=None; continue\n            cands.sort(key=lambda x:x[0]); d0, chosen=cands[0]\n            n.cluster_id=chosen.nid\n            e_ctrl=e_tx(CTRL_PACKET_BITS, d0)\n            if n.energy>=e_ctrl:\n                n.energy-=e_ctrl; self.total_energy_used+=e_ctrl; self.total_control_bits+=CTRL_PACKET_BITS\n                if chosen.energy>=e_rx(CTRL_PACKET_BITS):\n                    chosen.energy-=e_rx(CTRL_PACKET_BITS); self.total_energy_used+=e_rx(CTRL_PACKET_BITS)\n                else: chosen.alive=False\n            else:\n                n.alive=False; continue\n            self.clusters[chosen.nid].members.append(n.nid)\n\n    def transmit_round(self):\n        alive=self.alive_nodes(); chs=[n for n in alive if n.is_ch]\n        if not chs: return\n        data_events={}; expected_by_ch={ch.nid:set() for ch in chs}\n        for cl in self.clusters.values():\n            ch=self.nodes[cl.ch_id]\n            if not ch.alive: continue\n            for mid in cl.members:\n                m=self.nodes[mid]\n                if not m.alive: continue\n                allow=self.algo.allow_member_redundancy(m, ch)\n                rec=self._member_send(m, ch, allow, chs)\n                if mid not in data_events: data_events[mid]={\'tx_paths\':[], \'delivered\':False,\'timely\':False}\n                for rid in rec:\n                    e_cost=e_tx(DATA_PACKET_BITS, dist(m.pos(), self.nodes[rid].pos()))\n                    data_events[mid][\'tx_paths\'].append((e_cost, rid))\n                    expected_by_ch[rid].add(mid)\n        for cl in self.clusters.values():\n            ch=self.nodes[cl.ch_id]\n            if not ch.alive: continue\n            expected=set(cl.members); got=expected_by_ch[ch.nid]; missing=expected-got\n            for _ in missing:\n                ch.observed_fail+=0.1; self.total_drop+=1\n            for _ in got:\n                ch.observed_success+=0.3\n        delivered_by_ch={}\n        for ch in chs:\n            if not ch.alive: continue\n            n_recv=ch.queue_level\n            if n_recv>0:\n                e_aggr=E_DA*DATA_PACKET_BITS*n_recv\n                if ch.energy>=e_aggr: ch.energy-=e_aggr; self.total_energy_used+=e_aggr\n                else: ch.alive=False; continue\n            do_drop=False; do_delay=False\n            if ch.node_type=="malicious":\n                r=random.random()\n                if r<P_MAL_CH_DROP: do_drop=True\n                elif r<P_MAL_CH_DROP+P_MAL_CH_DELAY: do_delay=True\n            ok=False; timely=False; hops=0\n            if do_drop:\n                self.malicious_drop_packets+=n_recv; self.total_drop+=n_recv\n                ch.observed_fail+=1.0; ch.suspicion=min(1.0, ch.suspicion+0.3); ch.consecutive_strikes+=1\n            else:\n                relay,_meta=self.algo.choose_ch_relay(ch, chs)\n                if relay is not None and relay.alive:\n                    d1=dist(ch.pos(), relay.pos()); e1=e_tx(DATA_PACKET_BITS, d1)\n                    if ch.energy>=e1 and relay.energy>=e_rx(DATA_PACKET_BITS):\n                        ch.energy-=e1; self.total_energy_used+=e1\n                        relay.energy-=e_rx(DATA_PACKET_BITS); self.total_energy_used+=e_rx(DATA_PACKET_BITS)\n                        relay.queue_level += n_recv\n                        d2=dist(relay.pos(), self.bs); e2=e_tx(DATA_PACKET_BITS, d2)\n                        if relay.energy>=e2:\n                            relay.energy-=e2; self.total_energy_used+=e2\n                            ok=True; timely=(not do_delay); hops=2\n                            if do_delay: self.malicious_delay_times+=1; ch.observed_fail+=0.5\n                            else: ch.observed_success+=1.0; ch.consecutive_strikes=0\n                        else:\n                            relay.alive=False; self.total_drop+=n_recv\n                    else:\n                        ch.alive=False; self.total_drop+=n_recv\n                else:\n                    d_bs=dist(ch.pos(), self.bs); e_ch=e_tx(DATA_PACKET_BITS, d_bs)\n                    if ch.energy>=e_ch:\n                        ch.energy-=e_ch; self.total_energy_used+=e_ch\n                        ok=True; timely=(not do_delay); hops=1\n                        if do_delay: self.malicious_delay_times+=1; ch.observed_fail+=0.5\n                        else: ch.observed_success+=1.0; ch.consecutive_strikes=0\n                    else:\n                        ch.alive=False; self.total_drop+=n_recv\n            delivered_by_ch[ch.nid]=(ok,timely,hops)\n            ch.last_queue_level=ch.queue_level\n            self.algo.apply_watchdog(ch, ok, timely, chs)\n        for mid,info in data_events.items():\n            delivered=[]; timely_flag=False; hop_used=0\n            for e_cost, ch_id in info[\'tx_paths\']:\n                ok,t,h = delivered_by_ch.get(ch_id,(False,False,0))\n                if ok:\n                    delivered.append(e_cost)\n                    timely_flag = timely_flag or t\n                    hop_used = max(hop_used, h)\n            if delivered:\n                info[\'delivered\']=True; info[\'timely\']=timely_flag\n                self.total_delivered+=1; self.total_hop_count += (hop_used if hop_used>0 else 1)\n                if timely_flag: self.total_timely_delivered+=1\n                self.member_energy_effective += min(delivered)\n        self.algo.finalize_trust_blacklist()\n\n    def _member_send(self, m:Node, ch:Node, allow_redundancy:bool, chs:List[Node]):\n        rec=[]\n        if m.node_type=="malicious" and random.random()<P_MAL_MEMBER_DROP:\n            self.total_drop+=1; return rec\n        d=dist(m.pos(), ch.pos()); e=e_tx(DATA_PACKET_BITS, d)\n        if m.energy>=e:\n            m.energy-=e; self.total_energy_used+=e\n            if ch.energy>=e_rx(DATA_PACKET_BITS):\n                ch.energy-=e_rx(DATA_PACKET_BITS); self.total_energy_used+=e_rx(DATA_PACKET_BITS)\n                rec.append(ch.nid); ch.queue_level+=1\n            else: ch.alive=False\n        else:\n            m.alive=False; self.total_drop+=1; return rec\n        if allow_redundancy and chs:\n            alts=[]\n            for other in chs:\n                if other.nid==ch.nid or (not other.alive) or other.trust()<self.algo.trust_blacklist: continue\n                dd=dist(m.pos(), other.pos())\n                if dd<=COMM_RANGE: alts.append((dd,other))\n            if alts:\n                alts.sort(key=lambda x:x[0]); dd, alt=alts[0]\n                ee=e_tx(DATA_PACKET_BITS, dd)\n                if m.energy>=ee:\n                    m.energy-=ee; self.total_energy_used+=ee\n                    if alt.energy>=e_rx(DATA_PACKET_BITS):\n                        alt.energy-=e_rx(DATA_PACKET_BITS); self.total_energy_used+=e_rx(DATA_PACKET_BITS)\n                        rec.append(alt.nid); alt.queue_level+=1\n                    else: alt.alive=False\n        self.member_energy_total += e * (2 if (allow_redundancy and len(rec)>=2) else 1)\n        return rec\n\n    def update_lifetime(self):\n        dead = sum(1 for n in self.nodes if not n.alive)\n        if self.FND is None and dead>=1: self.FND=self.round\n        if self.HND is None and dead>=self.n_nodes//2: self.HND=self.round\n        if self.LND is None and dead>=self.n_nodes: self.LND=self.round\n\n    def step(self):\n        self.round += 1\n        if len(self.alive_nodes())==0:\n            self.update_lifetime(); return False\n        self.elect_cluster_heads(); self.assign_members(); self.transmit_round(); self.update_lifetime()\n        alive_cnt=sum(1 for n in self.nodes if n.alive)\n        ch_cnt=sum(1 for n in self.nodes if n.alive and n.is_ch)\n        timely=(self.total_timely_delivered / max(self.total_delivered,1))\n        self.sum_ch_count += ch_cnt\n        if len(self.clusters)>0:\n            import numpy as _np\n            csize = _np.mean([len(cl.members) for cl in self.clusters.values()])\n            self.sum_cluster_size += csize; self.count_cluster_rounds += 1\n        self.history.append({\'round\':self.round,\'alive\':alive_cnt,\'chs\':ch_cnt,\n                             \'cum_delivered\':self.total_delivered,\'cum_drop\':self.total_drop,\n                             \'cum_malicious_delay\':self.malicious_delay_times,\n                             \'cum_malicious_drop\':self.malicious_drop_packets,\n                             \'cum_timely_rate\':timely})\n        return (alive_cnt>0)\n\n    def run(self, rounds:int=SIM_ROUNDS):\n        initE=sum(n.energy for n in self.nodes)\n        if rounds is None or rounds<=0:\n            while self.step():\n                pass\n        else:\n            for _ in range(rounds):\n                if not self.step(): break\n        finalE=sum(n.energy for n in self.nodes)\n        self.total_energy_used += (initE-finalE)\n        bl_mal=sum(1 for n in self.nodes if (n.node_type=="malicious" and n.blacklisted))\n        bl_norm=sum(1 for n in self.nodes if (n.node_type=="normal"   and n.blacklisted))\n        energy_rate = self.member_energy_effective / max(self.member_energy_total,1.0)\n        timely_rate = self.total_timely_delivered / max(self.total_delivered,1)\n        pdr = self.total_delivered / max(self.total_delivered + self.total_drop, 1)\n        avg_ch = self.sum_ch_count / max(len(self.history),1)\n        avg_cluster = self.sum_cluster_size / max(self.count_cluster_rounds,1)\n        avg_hops = self.total_hop_count / max(self.total_delivered,1)\n        energy_per_del = self.total_energy_used / max(self.total_delivered,1)\n        throughput = (self.total_delivered*DATA_PACKET_BITS) / max(len(self.history),1)\n        return ({\n            \'algo\': self.algo.name,\n            \'FND\': self.FND if self.FND is not None else self.round,\n            \'HND\': self.HND if self.HND is not None else self.round,\n            \'LND\': self.LND if self.LND is not None else self.round,\n            \'drop_p\': int(self.total_drop),\n            \'total_p\': int(self.total_delivered),\n            \'pdr\': float(pdr),\n            \'timely_transfer_rate\': float(timely_rate),\n            \'energy_rate\': float(energy_rate),\n            \'energy_per_delivered\': float(energy_per_del),\n            \'throughput_bits_per_round\': float(throughput),\n            \'avg_ch_per_round\': float(avg_ch),\n            \'avg_cluster_size\': float(avg_cluster),\n            \'avg_hops_to_bs\': float(avg_hops),\n            \'malicious_delay\': int(self.malicious_delay_times),\n            \'malicious_drop\': int(self.malicious_drop_packets),\n            \'blacklisted_malicious\': int(bl_mal),\n            \'blacklisted_normal\': int(bl_norm),\n            \'control_overhead_bits\': int(self.total_control_bits),\n            \'rounds_run\': self.round\n        }, pd.DataFrame(self.history))\n', 'algs/sheer.py': '\n# -*- coding: utf-8 -*-\nfrom typing import List\nimport random, numpy as np\nfrom core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, e_tx, e_rx, clamp,\\\n                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS\n\ndef tri(x, a, b, c):\n    if x<=a or x>=c: return 0.0\n    if x==b: return 1.0\n    return (x-a)/(b-a) if x<a or x<b else (c-x)/(c-b)\n\nclass SHEER(AlgorithmBase):\n    @property\n    def name(self): return "SHEER-2025"\n    @property\n    def trust_warn(self): return 0.65\n    @property\n    def trust_blacklist(self): return 0.25\n    @property\n    def forget(self): return 0.98\n    @property\n    def strike_threshold(self): return 3\n\n    def _fis_score(self, e_norm, t, b_norm):\n        eL,eM,eH = tri(e_norm,0.0,0.0,0.5), tri(e_norm,0.25,0.5,0.75), tri(e_norm,0.5,1.0,1.0)\n        tL,tM,tH = tri(t,0.0,0.0,0.5), tri(t,0.25,0.5,0.75), tri(t,0.5,1.0,1.0)\n        bN,bM,bF = tri(b_norm,0.5,1.0,1.0), tri(b_norm,0.25,0.5,0.75), tri(b_norm,0.0,0.0,0.5)\n        rules = []\n        rules += [max(eH,tH,bN)*0.95, max(eH,tM,bN)*0.9, max(eM,tH,bN)*0.9]\n        rules += [max(eM,tM,bM)*0.7, max(eH,tM,bM)*0.8, max(eM,tH,bM)*0.8]\n        rules += [max(eL,tL,bF)*0.2, max(eL,tM,bF)*0.4, max(eM,tL,bF)*0.4]\n        num = sum(rules); den = len(rules) if len(rules)>0 else 1\n        return num/den\n\n    def select_cluster_heads(self):\n        sim=self.sim\n        alive=[n for n in sim.alive_nodes() if not n.blacklisted]\n        if not alive: return\n        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)\n        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)\n        for n in alive:\n            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue\n            e_norm=(n.energy - e_min)/(e_max-e_min + 1e-9)\n            t=n.trust()\n            b=1.0 - (dist(n.pos(), sim.bs)-d_min)/(d_max-d_min + 1e-9)\n            score = self._fis_score(e_norm, t, b)\n            p=clamp(BASE_P_CH*(0.5 + 0.8*score), 0.0, 1.0)\n            if random.random()<p:\n                n.is_ch=True; n.last_ch_round=sim.round\n                sim.clusters[n.nid]=Cluster(ch_id=n.nid)\n        if len(sim.clusters)<max(1, int(0.03*len(alive))):\n            cand=sorted([n for n in alive if not n.is_ch],\n                        key=lambda x: self._fis_score((x.energy-e_min)/(e_max-e_min+1e-9), x.trust(),\n                                                      1.0-(dist(x.pos(), sim.bs)-d_min)/(d_max-d_min+1e-9)),\n                        reverse=True)\n            for c in cand[:max(1,int(0.03*len(alive)))-len(sim.clusters)]:\n                c.is_ch=True; c.last_ch_round=sim.round; sim.clusters[c.nid]=Cluster(c.nid)\n\n    def allow_member_redundancy(self, member:Node, ch:Node)->bool:\n        return False\n\n    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):\n        sim=self.sim\n        d_bs=dist(ch.pos(), sim.bs)\n        best=None; best_cost=1e9; best_pair=(None,None)\n        for other in ch_nodes:\n            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue\n            d1=dist(ch.pos(), other.pos())\n            if d1>CH_NEIGHBOR_RANGE: continue\n            d2=dist(other.pos(), sim.bs)\n            cost=0.35*(d1+d2)+0.35*(1.0-other.trust())+0.30*(other.queue_level+1)\n            if cost<best_cost:\n                best_cost=cost; best=other; best_pair=(d1,d2)\n        use = (best is not None) and ((best_pair[0]+best_pair[1]) < 0.90*d_bs or ch.trust()<self.trust_warn)\n        return (best if use else None), ({\'d1\':best_pair[0],\'d2\':best_pair[1]} if use else {} )\n\n    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):\n        pass\n\n    def finalize_trust_blacklist(self):\n        sim=self.sim\n        for n in sim.alive_nodes():\n            n.trust_s = n.trust_s*self.forget + n.observed_success\n            n.trust_f = n.trust_f*self.forget + n.observed_fail\n            if n.trust()<self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:\n                n.blacklisted=True\n', 'algs/actar.py': '\n# -*- coding: utf-8 -*-\nfrom typing import List\nimport random, numpy as np\nfrom core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\\\n                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE\n\nclass ACTAR(AlgorithmBase):\n    @property\n    def name(self): return "ACTAR-2024"\n    @property\n    def trust_warn(self): return 0.60\n    @property\n    def trust_blacklist(self): return 0.26\n    @property\n    def forget(self): return 0.995\n    @property\n    def strike_threshold(self): return 4\n\n    def select_cluster_heads(self):\n        sim=self.sim\n        alive=[n for n in sim.alive_nodes() if not n.blacklisted]\n        if not alive: return\n        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)\n        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)\n        q_levels=[n.last_queue_level for n in alive]; q_min, q_max=min(q_levels), max(q_levels)\n        for n in alive:\n            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue\n            d_norm=(dist(n.pos(), sim.bs)-d_min)/(d_max-d_min+1e-9)\n            ring_scale = 1.20 - 0.40*d_norm\n            e=(n.energy-e_min)/(e_max-e_min+1e-9)\n            t=n.trust()\n            b=1.0 - d_norm\n            prox=1.0 - (n.last_queue_level-q_min)/(q_max-q_min+1e-9)\n            score=0.35*e + 0.30*t + 0.20*b + 0.15*prox\n            p=clamp(BASE_P_CH*ring_scale*(0.5+score),0.0,1.0)\n            if random.random()<p:\n                n.is_ch=True; n.last_ch_round=sim.round\n                sim.clusters[n.nid]=Cluster(n.nid)\n\n    def allow_member_redundancy(self, member:Node, ch:Node)->bool:\n        return False\n\n    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):\n        sim=self.sim\n        d_bs=dist(ch.pos(), sim.bs)\n        best=None; best_sum=1e9; pair=(None,None)\n        for other in ch_nodes:\n            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue\n            d1=dist(ch.pos(), other.pos()); \n            if d1>CH_NEIGHBOR_RANGE: continue\n            d2=dist(other.pos(), sim.bs)\n            if d1+d2<best_sum: best_sum=d1+d2; best=other; pair=(d1,d2)\n        if best is not None and best_sum < 0.9*d_bs and best.trust()>=self.trust_warn:\n            return best, {\'d1\':pair[0],\'d2\':pair[1]}\n        return None, {}\n\n    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):\n        if ok and timely: \n            if random.random()<0.03: ch.observed_fail += 0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)\n        else:\n            if random.random()<0.60: ch.observed_fail += 0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)\n\n    def finalize_trust_blacklist(self):\n        sim=self.sim\n        for n in sim.alive_nodes():\n            n.trust_s = n.trust_s*self.forget + n.observed_success\n            n.trust_f = n.trust_f*self.forget + n.observed_fail\n            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:\n                n.blacklisted = True\n            if n.suspicion >= 0.65: n.blacklisted=True\n', 'algs/tfsm_dpc.py': '\n# -*- coding: utf-8 -*-\nfrom typing import List\nimport random, numpy as np\nfrom core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\\\n                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE\n\nclass TFSM_DPC(AlgorithmBase):\n    @property\n    def name(self): return "TFSM-DPC-2024"\n    @property\n    def trust_warn(self): return 0.60\n    @property\n    def trust_blacklist(self): return 0.26\n    @property\n    def forget(self): return 0.995\n    @property\n    def strike_threshold(self): return 3\n\n    def _density_peaks(self, nodes:List[Node]):\n        coords=np.array([(n.x,n.y) for n in nodes])\n        dc=COMM_RANGE/2.0\n        dists=np.sqrt(((coords[:,None,:]-coords[None,:,:])**2).sum(-1))\n        rho=np.sum(np.exp(-(dists/dc)**2), axis=1) - 1.0\n        delta=np.zeros(len(nodes))\n        for i in range(len(nodes)):\n            higher=np.where(rho>rho[i])[0]\n            if len(higher)==0: delta[i]=np.max(dists[i,:])\n            else: delta[i]=np.min(dists[i,higher])\n        gamma=rho*delta\n        return rho, delta, gamma\n\n    def select_cluster_heads(self):\n        sim=self.sim\n        alive=[n for n in sim.alive_nodes() if not n.blacklisted]\n        if not alive: return\n        rho, delta, gamma = self._density_peaks(alive)\n        energies=np.array([n.energy for n in alive]); e_min,e_max=np.min(energies), np.max(energies)\n        e_norm=(energies - e_min)/(e_max-e_min + 1e-9)\n        t=np.array([n.trust() for n in alive])\n        gamma_ = 0.6*gamma + 0.2*e_norm + 0.2*t\n        K=max(1, int(BASE_P_CH*len(alive)))\n        idxs=np.argsort(-gamma_)[:K]\n        for idx in idxs:\n            n=alive[idx]\n            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue\n            n.is_ch=True; n.last_ch_round=sim.round\n            sim.clusters[n.nid]=Cluster(n.nid)\n        if len(sim.clusters)<K:\n            order=np.argsort(-rho)\n            for idx in order:\n                n=alive[idx]\n                if n.nid in sim.clusters: continue\n                n.is_ch=True; n.last_ch_round=sim.round; sim.clusters[n.nid]=Cluster(n.nid)\n                if len(sim.clusters)>=K: break\n\n    def allow_member_redundancy(self, member:Node, ch:Node)->bool:\n        return (self.trust_blacklist <= ch.trust() < self.trust_warn) and (ch.queue_level>0)\n\n    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):\n        sim=self.sim\n        d_bs=dist(ch.pos(), sim.bs)\n        best=None; best_score=1e18; pair=(None,None)\n        for other in ch_nodes:\n            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue\n            d1=dist(ch.pos(), other.pos())\n            if d1>CH_NEIGHBOR_RANGE: continue\n            d2=dist(other.pos(), sim.bs)\n            score=0.30*(d1+d2)+0.35*(other.queue_level+1)+0.35*(1.0-other.trust())\n            if score<best_score: best_score=score; best=other; pair=(d1,d2)\n        if best is not None and (pair[0]+pair[1])<0.9*d_bs:\n            return best, {\'d1\':pair[0],\'d2\':pair[1]}\n        return None, {}\n\n    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):\n        if ok and timely:\n            if random.random()<0.03: ch.observed_fail+=0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)\n        else:\n            if random.random()<0.65: ch.observed_fail+=0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)\n\n    def finalize_trust_blacklist(self):\n        sim=self.sim\n        for n in sim.alive_nodes():\n            n.trust_s = n.trust_s*self.forget + n.observed_success\n            n.trust_f = n.trust_f*self.forget + n.observed_fail\n            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:\n                n.blacklisted=True\n            if n.suspicion>=0.65: n.blacklisted=True\n', 'algs/dst_woa.py': '\n# -*- coding: utf-8 -*-\nfrom typing import List, Tuple\nimport random, math, numpy as np\nfrom core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\\\n                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE, DATA_PACKET_BITS, e_tx, e_rx\n\ndef ds_fuse(t:float, prior:float=0.5)->float:\n    m1_g, m1_b = t, 1.0-t\n    m2_g, m2_b = prior, 1.0-prior\n    K = m1_g*m2_b + m1_b*m2_g\n    if K >= 0.999999: return t\n    return (m1_g*m2_g) / (1.0 - K)\n\nclass DST_WOA(AlgorithmBase):\n    @property\n    def name(self): return "DST-WOA-2024"\n    @property\n    def trust_warn(self): return 0.60\n    @property\n    def trust_blacklist(self): return 0.26\n    @property\n    def forget(self): return 0.995\n    @property\n    def strike_threshold(self): return 3\n\n    def _evaluate_binary_chset(self, bitvec:np.ndarray)->float:\n        sim=self.sim\n        ch_idx=np.where(bitvec>0.5)[0]\n        if len(ch_idx)==0: return 1e12\n        ch_nodes=[sim.nodes[i] for i in ch_idx if (sim.nodes[i].alive and not sim.nodes[i].blacklisted)]\n        if not ch_nodes: return 1e12\n        cost_intra=0.0; penalty=0.0\n        for n in sim.nodes:\n            if n in ch_nodes: continue\n            dmin=1e9\n            for ch in ch_nodes:\n                d=dist(n.pos(), ch.pos()); dmin=min(dmin,d)\n            if dmin>COMM_RANGE: penalty+=1e6; continue\n            cost_intra += e_tx(DATA_PACKET_BITS, dmin) + e_rx(DATA_PACKET_BITS)*0.5\n        cost_up=sum(e_tx(DATA_PACKET_BITS, dist(ch.pos(), sim.bs)) for ch in ch_nodes)\n        trust_penalty=sum((1.0-ds_fuse(ch.trust(),0.5))*1e-4 for ch in ch_nodes)\n        return cost_intra + cost_up + penalty + trust_penalty\n\n    def _binary_woa(self, dim:int, K:int):\n        whales=12; iters=15; rnd=random.Random(42)\n        pop=np.clip(np.random.rand(whales, dim),0,1)\n        for i in range(whales):\n            idx=np.argsort(-pop[i])[:K]; vec=np.zeros(dim); vec[idx]=1.0; pop[i]=vec\n        fitness=np.array([self._evaluate_binary_chset(pop[i]) for i in range(whales)])\n        bi=int(np.argmin(fitness)); best=pop[bi].copy(); best_fit=float(fitness[bi])\n        for t in range(iters):\n            a=2 - 2*t/iters\n            for i in range(whales):\n                r1, r2 = rnd.random(), rnd.random()\n                A = 2*a*r1 - a; C = 2*r2\n                p = rnd.random()\n                if p<0.5:\n                    if abs(A)>=1:\n                        j=rnd.randrange(whales); D=np.abs(C*pop[j]-pop[i]); new=pop[j]-A*D\n                    else:\n                        D=np.abs(C*best - pop[i]); new=best - A*D\n                else:\n                    b=1; l=(rnd.random()*2-1); D=np.abs(best - pop[i]); new = D*np.e**(b*l)*np.cos(2*np.pi*l)+best\n                s=1/(1+np.exp(-new)); idx=np.argsort(-s)[:K]; vec=np.zeros(dim); vec[idx]=1.0; pop[i]=vec\n            fitness=np.array([self._evaluate_binary_chset(pop[i]) for i in range(whales)])\n            j=int(np.argmin(fitness))\n            if fitness[j] < best_fit: best_fit=float(fitness[j]); best=pop[j].copy()\n        return best\n\n    def select_cluster_heads(self):\n        sim=self.sim\n        alive=[n for n in sim.alive_nodes() if not n.blacklisted]\n        if not alive: return\n        dim=len(sim.nodes); K=max(1, int(BASE_P_CH*len(alive)))\n        bitvec=self._binary_woa(dim, K)\n        for i,v in enumerate(bitvec):\n            if v>0.5 and (sim.nodes[i] in alive) and ((sim.round - sim.nodes[i].last_ch_round) >= CH_COOLDOWN):\n                sim.nodes[i].is_ch=True; sim.nodes[i].last_ch_round=sim.round\n                sim.clusters[sim.nodes[i].nid]=Cluster(sim.nodes[i].nid)\n\n    def allow_member_redundancy(self, member:Node, ch:Node)->bool:\n        return False\n\n    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):\n        sim=self.sim\n        d_bs=dist(ch.pos(), sim.bs)\n        best=None; best_score=1e18; pair=(None,None)\n        for other in ch_nodes:\n            if other.nid==ch.nid or (not other.alive): continue\n            d1=dist(ch.pos(), other.pos()); \n            if d1>CH_NEIGHBOR_RANGE: continue\n            d2=dist(other.pos(), sim.bs)\n            t_fused = ds_fuse(other.trust(), 0.5)\n            score=0.30*(d1+d2)+0.45*(1.0-t_fused)+0.15*(other.queue_level+1)+0.10*(1.0-(other.energy/sim.nodes[0].energy))\n            if score<best_score: best_score=score; best=other; pair=(d1,d2)\n        if best is not None and (pair[0]+pair[1])<0.9*d_bs:\n            return best, {\'d1\':pair[0],\'d2\':pair[1]}\n        return None, {}\n\n    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):\n        if ok and timely:\n            if random.random()<0.03: ch.observed_fail+=0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)\n        else:\n            if random.random()<0.70: ch.observed_fail+=0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)\n\n    def finalize_trust_blacklist(self):\n        sim=self.sim\n        for n in sim.alive_nodes():\n            n.trust_s = n.trust_s*self.forget + n.observed_success\n            n.trust_f = n.trust_f*self.forget + n.observed_fail\n            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:\n                n.blacklisted=True\n            if n.suspicion>=0.65: n.blacklisted=True\n', 'algs/trail.py': '\n# -*- coding: utf-8 -*-\nfrom typing import List\nimport random, numpy as np\nfrom core.wsn_core import AlgorithmBase, Simulation, Node, Cluster, dist, clamp,\\\n                          BASE_P_CH, CH_COOLDOWN, COMM_RANGE, CH_NEIGHBOR_RANGE\n\nclass TRAIL(AlgorithmBase):\n    @property\n    def name(self): return "TRAIL (ours)"\n    @property\n    def trust_warn(self): return 0.55\n    @property\n    def trust_blacklist(self): return 0.35\n    @property\n    def forget(self): return 0.93\n    @property\n    def strike_threshold(self): return 2\n\n    def __init__(self, sim:\'Simulation\'):\n        super().__init__(sim)\n        self.rt_ratio=0.35; self.epsilon=0.15\n\n    def select_cluster_heads(self):\n        sim=self.sim\n        alive=[n for n in sim.alive_nodes() if not n.blacklisted]\n        if not alive: return\n        energies=[n.energy for n in alive]; e_min, e_max=min(energies), max(energies)\n        dists=[dist(n.pos(), sim.bs) for n in alive]; d_min, d_max=min(dists), max(dists)\n        q_levels=[n.last_queue_level for n in alive]; q_min, q_max=min(q_levels), max(q_levels)\n        for n in alive:\n            if (sim.round - n.last_ch_round) < CH_COOLDOWN: continue\n            e=(n.energy-e_min)/(e_max-e_min+1e-9)\n            t=n.trust()\n            q=1.0 - (n.last_queue_level - q_min)/(q_max - q_min + 1e-9)\n            b=1.0 - (dist(n.pos(), sim.bs)-d_min)/(d_max-d_min+1e-9)\n            s_pen = 1.0 - max(0.0, min(1.0, 1.0-n.suspicion))\n            score=0.35*e + 0.35*(t - 0.2*s_pen) + 0.20*q + 0.10*b\n            p=clamp(BASE_P_CH*(0.5+score),0.0,1.0)\n            if random.random()<p:\n                n.is_ch=True; n.last_ch_round=sim.round\n                sim.clusters[n.nid]=Cluster(n.nid)\n\n    def allow_member_redundancy(self, member:Node, ch:Node)->bool:\n        is_rt = (random.random()<self.rt_ratio)\n        return is_rt and (self.trust_blacklist <= ch.trust() < self.trust_warn)\n\n    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):\n        import math\n        sim=self.sim\n        d_bs=dist(ch.pos(), sim.bs); cost_direct=d_bs\n        cands=[]\n        for other in ch_nodes:\n            if other.nid==ch.nid or (not other.alive) or other.trust()<self.trust_blacklist: continue\n            d1=dist(ch.pos(), other.pos())\n            if d1>CH_NEIGHBOR_RANGE: continue\n            d2=dist(other.pos(), sim.bs)\n            score=0.35*(d1+d2)+0.30*(other.queue_level+1)+0.25*(1.0-other.trust())+0.10*(1.0-(other.energy/ (sim.nodes[0].energy if len(sim.nodes)>0 else 0.5)))\n            cands.append((score, other, d1, d2))\n        if not cands: return None, {}\n        cands.sort(key=lambda x:x[0])\n        best=cands[0]; relay=best[1]; d1,b_d2=best[2],best[3]\n        explore = (random.random()<self.epsilon)\n        if explore:\n            use_relay = (random.random()<0.5)\n        else:\n            use_relay = (d1+b_d2+1e-6) < 0.95*cost_direct\n        return (relay if use_relay else None), {\'d1\':d1,\'d2\':b_d2} if use_relay else {}\n\n    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]):\n        neigh=[]\n        for other in ch_nodes:\n            if other.nid==ch.nid or (not other.alive): continue\n            neigh.append((dist(ch.pos(), other.pos()), other))\n        neigh.sort(key=lambda x:x[0]); watchers=[p[1] for p in neigh[:2]]\n        for w in watchers:\n            if ok and timely:\n                if random.random()<0.05:\n                    ch.observed_fail += 0.2; ch.suspicion=min(1.0, ch.suspicion+0.1)\n            else:\n                if random.random()<0.75:\n                    ch.observed_fail += 0.7; ch.suspicion=min(1.0, ch.suspicion+0.3)\n\n    def finalize_trust_blacklist(self):\n        sim=self.sim\n        for n in sim.alive_nodes():\n            n.trust_s = n.trust_s*self.forget + n.observed_success\n            n.trust_f = n.trust_f*self.forget + n.observed_fail + 0.3*n.suspicion\n            if n.trust() < self.trust_blacklist or n.consecutive_strikes>=self.strike_threshold:\n                n.blacklisted=True\n'}
#
# def materialize(force=False):
#     for rel, code in FILES.items():
#         abspath = os.path.join(os.path.dirname(__file__), rel)
#         os.makedirs(os.path.dirname(abspath), exist_ok=True)
#         if force or (not os.path.exists(abspath)):
#             with open(abspath, "w", encoding="utf-8") as f:
#                 f.write(code)

ALGOS={'sheer': ('algs.sheer','SHEER'),'actar': ('algs.actar','ACTAR'),
       'tfsm_dpc': ('algs.tfsm_dpc','TFSM_DPC'),'dst_woa': ('algs.dst_woa','DST_WOA'),
       'trail': ('algs.trail','TRAIL')}

def get_algo_ctor(tag:str):
    mod_name, cls = ALGOS[tag]
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls)

def run_one(tag:str, n_nodes:int, n_malicious:int, seed:int, rounds:int, until_dead:bool, out_dir:str, suffix:str=""):
    from core.wsn_core import Simulation
    ctor = get_algo_ctor(tag)
    sim = Simulation(ctor, n_nodes=n_nodes, n_malicious=n_malicious, seed=seed)
    if until_dead:
        res, hist = sim.run(rounds=0)  # 0 => run until LND
    else:
        res, hist = sim.run(rounds=rounds)
    if suffix:
        hist.to_csv(os.path.join(out_dir, f'hist_{suffix}_{tag}.csv'), index=False)
    else:
        hist.to_csv(os.path.join(out_dir, f'hist_{tag}.csv'), index=False)
    res['seed']=seed
    return res

def run_all_tags(tags, n_nodes:int, n_malicious:int, seeds:list, rounds:int, until_dead:bool, out_dir:str, suffix:str=""):
    os.makedirs(out_dir, exist_ok=True)
    rows=[]
    for seed in seeds:
        for tag in tags:
            print(f'Running {tag} (seed={seed}) ...')
            res = run_one(tag, n_nodes, n_malicious, seed, rounds, until_dead, out_dir, suffix=suffix)
            rows.append(res)
    df = pd.DataFrame(rows)
    df.insert(0, 'algo_tag', [r['algo'] for r in rows])
    if suffix:
        df.to_csv(os.path.join(out_dir, f'summary_{suffix}.csv'), index=False)
    else:
        df.to_csv(os.path.join(out_dir, f'summary.csv'), index=False)
    return df

# >>> CHANGED: 更稳的聚合方式（均值±标准差），顺序固定，误差线与数值标注
def plot_bars(summary: pd.DataFrame, out_dir: str, algo_order=None):
    if algo_order is None:
        algo_order = ["SHEER-2025", "ACTAR-2024", "TFSM-DPC-2024", "DST-WOA-2024", "TRAIL (ours)"]

    metrics = [
        ('pdr', 'Packet Delivery Ratio'),
        ('timely_transfer_rate', 'Timely Transfer Rate'),
        ('energy_per_delivered', 'Energy per Delivered (J/packet)'),
        ('throughput_bits_per_round', 'Throughput (bits/round)'),
        ('avg_hops_to_bs', 'Average Hops to BS'),
        ('drop_p', 'Total Drops'),
        ('FND', 'FND'),
        ('HND', 'HND'),
        ('LND', 'LND'),
    ]

    # 预聚合（按 algo ）
    grouped = summary.groupby('algo')
    means = grouped.mean(numeric_only=True)
    stds  = grouped.std(numeric_only=True).fillna(0.0)
    counts = grouped.size()

    # 算法显示顺序（只保留出现过的）
    algos = [a for a in algo_order if a in means.index]
    x = np.arange(len(algos))

    for key, title in metrics:
        if key not in means.columns:
            continue
        y = means.loc[algos, key].values
        yerr = stds.loc[algos, key].values if (counts.max() > 1) else None

        plt.figure(figsize=(8, 4.2))
        bars = plt.bar(x, y, yerr=yerr, capsize=4)
        plt.xticks(x, algos, rotation=20, ha='right')
        plt.title(title)
        plt.ylabel(key)
        plt.grid(axis='y', linestyle='--', alpha=0.3)

        # 数值标注
        for i, b in enumerate(bars):
            val = y[i]
            plt.text(b.get_x() + b.get_width()/2, b.get_height()*1.01,
                     f'{val:.3g}' if abs(val) < 1000 else f'{val:,.0f}',
                     ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'metric_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'metric_{key}.svg'), dpi=150)  # 矢量图
        plt.close()


# >>> CHANGED: 容错、网格、线宽、SVG 导出
def plot_timely_curves(tags, out_dir: str, suffix: str = ""):
    plt.figure(figsize=(8, 4.2))
    for tag in tags:
        fname = f'hist_{suffix}_{tag}.csv' if suffix else f'hist_{tag}.csv'
        fpath = os.path.join(out_dir, fname)
        if not os.path.exists(fpath):
            print(f"[warn] history file not found: {fpath}")
            continue
        h = pd.read_csv(fpath)
        plt.plot(h['round'], h['cum_timely_rate'], label=tag, linewidth=1.8)
    plt.xlabel('round')
    plt.ylabel('cumulative timely transfer rate')
    plt.legend()
    plt.title('Timely Transfer Rate (Cumulative)')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    name = f'timely_rate_curves_{suffix}.png' if suffix else 'timely_rate_curves.png'
    plt.savefig(os.path.join(out_dir, name), dpi=150)
    # 另存 SVG
    name_svg = f'timely_rate_curves_{suffix}.svg' if suffix else 'timely_rate_curves.svg'
    plt.savefig(os.path.join(out_dir, name_svg), dpi=150)
    plt.close()


def parse_ratios(s: str):
    items = [x.strip() for x in s.split(',') if x.strip()]
    vals = []
    for it in items:
        try:
            v = float(it)
            if 0.0 < v < 1.0:
                vals.append(v)
        except:
            pass
    return sorted(set(vals))

# >>> CHANGED: 每个算法用自己的 xs（防止缺某个 ratio 时长度不匹配），误差带对齐
def agg_mean_std(df_all: pd.DataFrame, key: str):
    g = df_all.groupby(['algo', 'ratio'])[key]
    mu = g.mean().reset_index(name='mean')
    sd = g.std().fillna(0.0).reset_index(name='std')
    cnt = g.count().reset_index(name='n')
    out = mu.merge(sd, on=['algo', 'ratio']).merge(cnt, on=['algo', 'ratio'])
    return out

def plot_lines_with_error(df_all: pd.DataFrame, out_dir: str, seeds: int):
    algo_order = ["SHEER-2025", "ACTAR-2024", "TFSM-DPC-2024", "DST-WOA-2024", "TRAIL (ours)"]
    metrics = [
        ('pdr', 'PDR (Packet Delivery Ratio)'),
        ('drop_rate', 'Drop Rate'),
        ('timely_transfer_rate', 'Timely Transfer Rate'),
        ('energy_per_delivered', 'Energy per Delivered Packet (J/packet)'),
        ('throughput_bits_per_round', 'Throughput (bits/round)'),
        ('avg_hops_to_bs', 'Average Hops to BS'),
        ('FND', 'FND (First Node Dies)'),
        ('HND', 'HND (Half Nodes Die)'),
        ('LND', 'LND (Last Node Dies)'),
    ]

    for key, title in metrics:
        plt.figure(figsize=(8.6, 4.6))
        for algo_name in algo_order:
            sub = df_all[df_all['algo'] == algo_name]
            if sub.empty:
                continue
            stats = agg_mean_std(sub, key).sort_values('ratio')
            if stats.empty:
                continue
            xs = (stats['ratio'] * 100).values   # 每个算法自己的 x（百分比）
            ys = stats['mean'].values
            plt.plot(xs, ys, marker='o', linewidth=1.8, label=algo_name)
            if seeds > 1:
                sd = stats['std'].values
                plt.fill_between(xs, ys - sd, ys + sd, alpha=0.18)

        plt.xlabel('Malicious ratio (%)')
        plt.ylabel(key)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'line_{key}.png'), dpi=150)
        plt.savefig(os.path.join(out_dir, f'line_{key}.svg'), dpi=150)
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, choices=list(ALGOS.keys()))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--ratios', type=str, help='例如 0.1,0.2,0.3,0.4,0.5')
    parser.add_argument('--rounds', type=int, default=500)
    parser.add_argument('--until-dead', action='store_true', help='忽略 --rounds，直至能量耗尽 (LND)')
    parser.add_argument('--seeds', type=int, default=1, help='重复实验次数（不同随机种子）')
    args = parser.parse_args()

    # materialize(force=args.force)

    out_dir= 'outputs'; os.makedirs(out_dir, exist_ok=True)
    tags = [args.algo] if args.algo else list(ALGOS.keys())
    if args.all or (not args.algo): tags=list(ALGOS.keys())

    if args.ratios:
        ratios = parse_ratios(args.ratios)
        if not ratios:
            print("ratios 解析失败，请使用例如 --ratios 0.1,0.2,0.3,0.4,0.5")
            sys.exit(1)
        all_rows=[]
        seeds_list=list(range(42, 42+args.seeds))
        for r in ratios:
            n_nodes=100; n_mal=int(n_nodes*r)
            df = run_all_tags(tags, n_nodes, n_mal, seeds_list, rounds=args.rounds,
                              until_dead=args.until_dead, out_dir=out_dir, suffix=f'ratio_{int(r*100)}')
            # 追加 drop_rate 派生列
            df['ratio'] = r
            # >>> CHANGED: 统一端到端口径
            df['drop_rate'] = 1.0 - df['pdr']
            all_rows.append(df)
            plot_timely_curves(tags, out_dir, suffix=f'ratio_{int(r*100)}')
        df_all = pd.concat(all_rows, ignore_index=True)
        df_all.to_csv(os.path.join(out_dir, 'summary_all.csv'), index=False)
        plot_lines_with_error(df_all, out_dir, seeds=args.seeds)
        print("批量完成：summary_all.csv 与 line_*.png 已生成。")
    else:
        seeds_list=list(range(42, 42+args.seeds))
        df = run_all_tags(tags, 100, 30, seeds_list, rounds=args.rounds,
                          until_dead=args.until_dead, out_dir=out_dir, suffix="")
        # >>> CHANGED: 统一端到端口径
        df['drop_rate'] = 1.0 - df['pdr']
        plot_bars(df, out_dir)
        plot_timely_curves(tags, out_dir)
        print("单次运行完成：summary.csv / metric_*.png / timely_rate_curves.png")

if __name__ == '__main__':
    main()
