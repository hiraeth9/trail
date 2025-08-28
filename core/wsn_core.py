
# -*- coding: utf-8 -*-
import math, random, os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

SEED = 42
AREA_W, AREA_H = 100.0, 100.0
BS_POS = (50.0, 150.0)
COMM_RANGE = 30.0
CH_NEIGHBOR_RANGE = 60.0
INIT_ENERGY = 0.5

E_ELEC = 50e-9; E_FS = 10e-12; E_MP = 0.0013e-12
D0 = math.sqrt(E_FS / E_MP); E_DA = 5e-9
DATA_PACKET_BITS = 4000; CTRL_PACKET_BITS = 200

SIM_ROUNDS = 500
BASE_P_CH = 0.07
CH_COOLDOWN = int(1.0/BASE_P_CH)

P_MAL_MEMBER_DROP = 0.25
P_MAL_CH_DROP = 0.60
P_MAL_CH_DELAY = 0.30

def dist(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    return math.hypot(a[0]-b[0], a[1]-b[1])

def e_tx(bits: int, d: float) -> float:
    return E_ELEC*bits + (E_FS*bits*(d**2) if d < D0 else E_MP*bits*(d**4))

def e_rx(bits: int) -> float:
    return E_ELEC*bits

def clamp(x, lo, hi): return max(lo, min(hi, x))

@dataclass
class Node:
    nid: int; x: float; y: float; node_type: str
    energy: float = INIT_ENERGY
    trust_s: float = 0.0; trust_f: float = 0.0
    suspicion: float = 0.0
    consecutive_strikes: int = 0
    blacklisted: bool = False
    last_ch_round: int = -9999
    alive: bool = True
    is_ch: bool = False; cluster_id: Optional[int] = None
    observed_success: float = 0.0; observed_fail: float = 0.0
    queue_level: int = 0; last_queue_level: int = 0
    dir_cnt: int = 0; dir_val: float = 0.0
    rly_cnt: int = 0; rly_val: float = 0.0
    def pos(self): return (self.x, self.y)
    def trust(self): return (self.trust_s + 1.0) / (self.trust_s + self.trust_f + 2.0)
    def reset_round_flags(self):
        self.is_ch=False; self.cluster_id=None
        self.observed_success=0.0; self.observed_fail=0.0
        self.queue_level=0; self.suspicion=max(0.0, self.suspicion*0.9)

@dataclass
class Cluster:
    ch_id: int; members: List[int] = field(default_factory=list)

class AlgorithmBase:
    def __init__(self, sim:'Simulation'): self.sim = sim
    @property
    def name(self)->str: return "BaseAlgo"
    @property
    def trust_warn(self)->float: raise NotImplementedError
    @property
    def trust_blacklist(self)->float: raise NotImplementedError
    @property
    def forget(self)->float: return 0.98
    @property
    def strike_threshold(self)->int: return 3
    def suspicion_blacklist(self)->Optional[float]: return None
    def select_cluster_heads(self): raise NotImplementedError
    def allow_member_redundancy(self, member:Node, ch:Node)->bool: return False
    def choose_ch_relay(self, ch:Node, ch_nodes:List[Node]):
        return None, {}
    def apply_watchdog(self, ch:Node, ok:bool, timely:bool, ch_nodes:List[Node]): pass
    def finalize_trust_blacklist(self): raise NotImplementedError

class Simulation:
    def __init__(self, algo_ctor, n_nodes:int=100, n_malicious:int=30, seed:int=SEED):
        self.n_nodes=n_nodes; self.n_malicious=n_malicious
        self.bs=BS_POS; self.round=0; self.seed=seed
        self.nodes: List[Node]=[]; self.clusters: Dict[int, Cluster]={}
        self.history=[]; self.rand=random.Random(seed)
        self._init_nodes()
        self.total_drop=0; self.total_delivered=0; self.total_timely_delivered=0
        self.malicious_delay_times=0; self.malicious_drop_packets=0
        self.member_energy_total=0.0; self.member_energy_effective=0.0
        self.total_energy_used=0.0; self.total_control_bits=0; self.total_hop_count=0
        self.sum_cluster_size=0; self.count_cluster_rounds=0; self.sum_ch_count=0
        self.FND=None; self.HND=None; self.LND=None
        self.algo: AlgorithmBase = algo_ctor(self)

    def _init_nodes(self):
        coords = np.column_stack([np.random.rand(self.n_nodes)*AREA_W,
                                  np.random.rand(self.n_nodes)*AREA_H])
        mal = set(self.rand.sample(range(self.n_nodes), self.n_malicious))
        for i in range(self.n_nodes):
            t = "malicious" if i in mal else "normal"
            self.nodes.append(Node(i, float(coords[i,0]), float(coords[i,1]), t))

    def alive_nodes(self): return [n for n in self.nodes if n.alive]

    def elect_cluster_heads(self):
        self.clusters={}
        for n in self.nodes: n.reset_round_flags()
        self.algo.select_cluster_heads()
        if len(self.clusters)==0:
            alive=[n for n in self.alive_nodes() if not n.blacklisted] or self.alive_nodes()
            if alive:
                ch=max(alive, key=lambda x:x.energy)
                ch.is_ch=True; ch.last_ch_round=self.round; self.clusters[ch.nid]=Cluster(ch.nid)

    def assign_members(self):
        alive=self.alive_nodes(); chs=[n for n in alive if n.is_ch]
        if not chs: return
        for n in alive:
            if n.is_ch: continue
            cands=[]
            for ch in chs:
                if ch.trust()<self.algo.trust_blacklist: continue
                d=dist(n.pos(), ch.pos())
                if d<=COMM_RANGE: cands.append((d,ch))
            if not cands: n.cluster_id=None; continue
            cands.sort(key=lambda x:x[0]); d0, chosen=cands[0]
            n.cluster_id=chosen.nid
            e_ctrl=e_tx(CTRL_PACKET_BITS, d0)
            if n.energy>=e_ctrl:
                n.energy-=e_ctrl; self.total_energy_used+=e_ctrl; self.total_control_bits+=CTRL_PACKET_BITS
                if chosen.energy>=e_rx(CTRL_PACKET_BITS):
                    chosen.energy-=e_rx(CTRL_PACKET_BITS); self.total_energy_used+=e_rx(CTRL_PACKET_BITS)
                else: chosen.alive=False
            else:
                n.alive=False; continue
            self.clusters[chosen.nid].members.append(n.nid)

    def transmit_round(self):
        alive = self.alive_nodes()
        chs = [n for n in alive if n.is_ch]
        if not chs:
            return

        # —— 1) 成员上行：记录每个成员的发包路径与期望到达的CH —— #
        data_events = {}
        expected_by_ch = {ch.nid: set() for ch in chs}
        for cl in self.clusters.values():
            ch = self.nodes[cl.ch_id]
            if not ch.alive:
                continue
            for mid in cl.members:
                m = self.nodes[mid]
                if not m.alive:
                    continue
                allow = self.algo.allow_member_redundancy(m, ch)
                rec = self._member_send(m, ch, allow, chs)
                if mid not in data_events:
                    data_events[mid] = {'tx_paths': [], 'delivered': False, 'timely': False}
                for rid in rec:
                    e_cost = e_tx(DATA_PACKET_BITS, dist(m.pos(), self.nodes[rid].pos()))
                    data_events[mid]['tx_paths'].append((e_cost, rid))
                    expected_by_ch[rid].add(mid)

        # —— 2) CH侧：统计“应到未到/已到”的观测 —— #
        for cl in self.clusters.values():
            ch = self.nodes[cl.ch_id]
            if not ch.alive:
                continue
            expected = set(cl.members)
            got = expected_by_ch[ch.nid]
            missing = expected - got
            for _ in missing:
                ch.observed_fail += 0.1
                self.total_drop += 1
            for _ in got:
                ch.observed_success += 0.3

        # —— 3) CH下行：聚合能耗、决定直达or两跳、能量记账、看门狗 —— #
        delivered_by_ch = {}
        for ch in chs:
            if not ch.alive:
                continue

            n_recv = ch.queue_level  # 本轮收到的成员数量（用于聚合与发包）
            used_relay = False  # 本轮是否采用两跳（供算法层调参/看门狗使用）
            ok = False
            timely = False
            hops = 0

            # 3.1 聚合处理能耗（有包才聚合）
            if n_recv > 0:
                e_aggr = E_DA * DATA_PACKET_BITS * n_recv
                if ch.energy >= e_aggr:
                    ch.energy -= e_aggr
                    self.total_energy_used += e_aggr
                else:
                    # 聚合能量都不够 → 本轮包都掉；但CH未必“死亡”，按能耗模型可以选择判死
                    ch.alive = False
                    self.total_drop += n_recv
                    delivered_by_ch[ch.nid] = (False, False, 0)
                    # 正确记录本轮负载并清空队列
                    ch.last_queue_level = n_recv
                    ch.queue_level = 0
                    self.algo.mode_by_ch[ch.nid] = 'direct'
                    self.algo.apply_watchdog(ch, False, False, chs)
                    continue

            # 3.2 恶意CH行为决定（丢弃/延迟）
            do_drop = False
            do_delay = False
            if ch.node_type == "malicious":
                r = random.random()
                if r < P_MAL_CH_DROP:
                    do_drop = True
                elif r < P_MAL_CH_DROP + P_MAL_CH_DELAY:
                    do_delay = True

            # 3.3 发包：先尝试两跳；若两跳的“首跳收发”条件不满足，回退直达
            if do_drop and n_recv > 0:
                # 恶意直接丢
                self.malicious_drop_packets += n_recv
                self.total_drop += n_recv
                ch.observed_fail += 1.0
                ch.suspicion = min(1.0, ch.suspicion + 0.3)
                ch.consecutive_strikes += 1
            else:
                relay, _meta = self.algo.choose_ch_relay(ch, chs)
                if relay is not None and relay.alive and n_recv > 0:
                    # —— 两跳：首跳（CH→relay）
                    d1 = dist(ch.pos(), relay.pos())
                    e1 = e_tx(DATA_PACKET_BITS, d1)
                    e_rx_once = e_rx(DATA_PACKET_BITS)
                    if ch.energy >= e1 and relay.energy >= e_rx_once:
                        ch.energy -= e1
                        self.total_energy_used += e1
                        relay.energy -= e_rx_once
                        self.total_energy_used += e_rx_once

                        # —— 两跳：次跳（relay→BS）
                        d2 = dist(relay.pos(), self.bs)
                        e2 = e_tx(DATA_PACKET_BITS, d2)
                        if relay.energy >= e2:
                            relay.energy -= e2
                            self.total_energy_used += e2
                            ok = True
                            timely = (not do_delay)
                            hops = 2
                            used_relay = True
                            if do_delay:
                                self.malicious_delay_times += 1
                                ch.observed_fail += 0.5
                            else:
                                ch.observed_success += 1.0
                                ch.consecutive_strikes = 0
                        else:
                            # 中继在次跳时没电了 → 包丢；中继死亡合理
                            relay.alive = False
                            self.total_drop += n_recv
                            ch.observed_fail += 0.6
                    else:
                        # ⚠ 首跳条件不满足：不要直接“杀CH”，先尝试改为直达
                        d_bs = dist(ch.pos(), self.bs)
                        e_ch = e_tx(DATA_PACKET_BITS, d_bs)
                        if ch.energy >= e_ch and n_recv > 0:
                            ch.energy -= e_ch
                            self.total_energy_used += e_ch
                            ok = True
                            timely = (not do_delay)
                            hops = 1
                            used_relay = False
                            if do_delay:
                                self.malicious_delay_times += 1
                                ch.observed_fail += 0.5
                            else:
                                ch.observed_success += 1.0
                                ch.consecutive_strikes = 0
                        else:
                            ch.alive = False
                            self.total_drop += n_recv
                            ch.observed_fail += 0.6
                else:
                    # —— 直达（无合适中继/没有数据/或中继已死）
                    if n_recv > 0:
                        d_bs = dist(ch.pos(), self.bs)
                        e_ch = e_tx(DATA_PACKET_BITS, d_bs)
                        if ch.energy >= e_ch:
                            ch.energy -= e_ch
                            self.total_energy_used += e_ch
                            ok = True
                            timely = (not do_delay)
                            hops = 1
                            used_relay = False
                            if do_delay:
                                self.malicious_delay_times += 1
                                ch.observed_fail += 0.5
                            else:
                                ch.observed_success += 1.0
                                ch.consecutive_strikes = 0
                        else:
                            ch.alive = False
                            self.total_drop += n_recv
                            ch.observed_fail += 0.6
                    else:
                        # 本轮无数据可发：保持 ok=False / hops=0
                        used_relay = False

            delivered_by_ch[ch.nid] = (ok, timely, hops)

            # 3.4 正确记录与重置队列（很关键！）
            ch.last_queue_level = n_recv
            ch.queue_level = 0

            # 3.5 告知算法层本轮选择，用于看门狗/自适应
            self.algo.mode_by_ch[ch.nid] = 'relay' if used_relay else 'direct'
            self.algo.apply_watchdog(ch, ok, timely, chs)

        # —— 4) 汇总成员下行结果（每个成员看自己所有上行路径里是否有成功的CH） —— #
        for mid, info in data_events.items():
            delivered = []
            timely_flag = False
            hop_used = 0
            for e_cost, ch_id in info['tx_paths']:
                ok, t, h = delivered_by_ch.get(ch_id, (False, False, 0))
                if ok:
                    delivered.append(e_cost)
                    timely_flag = timely_flag or t
                    hop_used = max(hop_used, h)
            if delivered:
                info['delivered'] = True
                info['timely'] = timely_flag
                self.total_delivered += 1
                self.total_hop_count += (hop_used if hop_used > 0 else 1)
                if timely_flag:
                    self.total_timely_delivered += 1
                self.member_energy_effective += min(delivered)

        # —— 5) 每轮结束：更新信任/黑名单 & 自适应策略（若算法实现了的话） —— #
        self.algo.finalize_trust_blacklist()

    def _member_send(self, m:Node, ch:Node, allow_redundancy:bool, chs:List[Node]):
        rec=[]
        if m.node_type=="malicious" and random.random()<P_MAL_MEMBER_DROP:
            self.total_drop+=1; return rec
        d=dist(m.pos(), ch.pos()); e=e_tx(DATA_PACKET_BITS, d)
        if m.energy>=e:
            m.energy-=e; self.total_energy_used+=e
            if ch.energy>=e_rx(DATA_PACKET_BITS):
                ch.energy-=e_rx(DATA_PACKET_BITS); self.total_energy_used+=e_rx(DATA_PACKET_BITS)
                rec.append(ch.nid); ch.queue_level+=1
            else: ch.alive=False
        else:
            m.alive=False; self.total_drop+=1; return rec
        if allow_redundancy and chs:
            alts=[]
            for other in chs:
                if other.nid==ch.nid or (not other.alive) or other.trust()<self.algo.trust_blacklist: continue
                dd=dist(m.pos(), other.pos())
                if dd<=COMM_RANGE: alts.append((dd,other))
            if alts:
                alts.sort(key=lambda x:x[0]); dd, alt=alts[0]
                ee=e_tx(DATA_PACKET_BITS, dd)
                if m.energy>=ee:
                    m.energy-=ee; self.total_energy_used+=ee
                    if alt.energy>=e_rx(DATA_PACKET_BITS):
                        alt.energy-=e_rx(DATA_PACKET_BITS); self.total_energy_used+=e_rx(DATA_PACKET_BITS)
                        rec.append(alt.nid); alt.queue_level+=1
                    else: alt.alive=False
        self.member_energy_total += e * (2 if (allow_redundancy and len(rec)>=2) else 1)
        return rec

    def update_lifetime(self):
        dead = sum(1 for n in self.nodes if not n.alive)
        if self.FND is None and dead>=1: self.FND=self.round
        if self.HND is None and dead>=self.n_nodes//2: self.HND=self.round
        if self.LND is None and dead>=self.n_nodes: self.LND=self.round

    def step(self):
        # 新增：打印轮次信息，只针对指定的5个算法
        target_algos = ["ACTAR-2024", "TRAIL (ours)", "DST-WOA-2024", "SHEER-2025", "TFSM-DPC-2024"]
        if self.algo.name in target_algos:
            print(f"Algorithm: {self.algo.name}, Round: {self.round}")
        self.round += 1

        if len(self.alive_nodes())==0:
            self.update_lifetime(); return False
        self.elect_cluster_heads(); self.assign_members(); self.transmit_round(); self.update_lifetime()
        alive_cnt=sum(1 for n in self.nodes if n.alive)
        ch_cnt=sum(1 for n in self.nodes if n.alive and n.is_ch)
        timely=(self.total_timely_delivered / max(self.total_delivered,1))
        self.sum_ch_count += ch_cnt
        if len(self.clusters)>0:
            import numpy as _np
            csize = _np.mean([len(cl.members) for cl in self.clusters.values()])
            self.sum_cluster_size += csize; self.count_cluster_rounds += 1
        self.history.append({'round':self.round,'alive':alive_cnt,'chs':ch_cnt,
                             'cum_delivered':self.total_delivered,'cum_drop':self.total_drop,
                             'cum_malicious_delay':self.malicious_delay_times,
                             'cum_malicious_drop':self.malicious_drop_packets,
                             'cum_timely_rate':timely})
        return (alive_cnt>0)

    def run(self, rounds:int=SIM_ROUNDS):
        initE=sum(n.energy for n in self.nodes)
        if rounds is None or rounds<=0:
            while self.step():
                pass
        else:
            for _ in range(rounds):
                if not self.step(): break
        finalE=sum(n.energy for n in self.nodes)
        self.total_energy_used += (initE-finalE)
        bl_mal=sum(1 for n in self.nodes if (n.node_type=="malicious" and n.blacklisted))
        bl_norm=sum(1 for n in self.nodes if (n.node_type=="normal"   and n.blacklisted))
        energy_rate = self.member_energy_effective / max(self.member_energy_total,1.0)
        timely_rate = self.total_timely_delivered / max(self.total_delivered,1)
        pdr = self.total_delivered / max(self.total_delivered + self.total_drop, 1)
        avg_ch = self.sum_ch_count / max(len(self.history),1)
        avg_cluster = self.sum_cluster_size / max(self.count_cluster_rounds,1)
        avg_hops = self.total_hop_count / max(self.total_delivered,1)
        energy_per_del = self.total_energy_used / max(self.total_delivered,1)
        throughput = (self.total_delivered*DATA_PACKET_BITS) / max(len(self.history),1)
        return ({
            'algo': self.algo.name,
            'FND': self.FND if self.FND is not None else self.round,
            'HND': self.HND if self.HND is not None else self.round,
            'LND': self.LND if self.LND is not None else self.round,
            'drop_p': int(self.total_drop),
            'total_p': int(self.total_delivered),
            'pdr': float(pdr),
            'timely_transfer_rate': float(timely_rate),
            'energy_rate': float(energy_rate),
            'energy_per_delivered': float(energy_per_del),
            'throughput_bits_per_round': float(throughput),
            'avg_ch_per_round': float(avg_ch),
            'avg_cluster_size': float(avg_cluster),
            'avg_hops_to_bs': float(avg_hops),
            'malicious_delay': int(self.malicious_delay_times),
            'malicious_drop': int(self.malicious_drop_packets),
            'blacklisted_malicious': int(bl_mal),
            'blacklisted_normal': int(bl_norm),
            'control_overhead_bits': int(self.total_control_bits),
            'rounds_run': self.round
        }, pd.DataFrame(self.history))
