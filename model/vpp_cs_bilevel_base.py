# -*- coding: utf-8 -*-
"""
终极版双层模型 — 基础模块
面向快充服务场景的、考虑电热耦合与不确定性的、VPP—储能型充电站双层协调优化模型

模块职责：
  1. 全局常量与参数类定义
  2. 电价/温度数据加载
  3. 电热耦合物理函数（温度→效率/SOC/功率折减）
  4. 充电需求生成与场景缩减
  5. 市场准入评估
  6. 绘图样式
"""
import os
from dataclasses import dataclass, field
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ════════════════════════════════════════════════════════════════
# 全局常量
# ════════════════════════════════════════════════════════════════
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXCEL_FILE = os.path.join(BASE_DIR, "guangdong.xlsx")

N = 96                  # 96个15分钟时段
DT = 0.25               # 时间步长(h)
HOURS = np.arange(N, dtype=float) * DT

N_STATIONS = 7
TRAFFIC_WEIGHTS = np.array([1.00, 0.90, 1.15, 0.95, 1.05, 0.85, 1.10])
FAST_SVC_HOURS = [8, 13, 19]       # 快充高峰时刻
N_SLOW_SAMPLES = 100                # 慢充蒙特卡洛样本数
N_SLOW_SCENARIOS = 5                # 缩减后场景数
BASE_SEED = 2026

SLOW_CAR_MEAN = np.array(
    [1, 1, 1, 1, 1, 2, 3, 4, 5, 6, 6, 5, 4, 4, 4, 5, 6, 7, 8, 8, 6, 4, 2, 1],
    dtype=float)

# 求解参数
BIG_M = 1e3                         # KKT互补松弛线性化Big-M
THROUGHPUT_COST = 0.01              # 储能吞吐成本(元/kWh)
FAIRNESS_CS_PENALTY = 1e-3          # 公平性偏差惩罚系数
KKT_DUAL_UB = 1e4                   # 对偶变量上界


# ════════════════════════════════════════════════════════════════
# 参数类
# ════════════════════════════════════════════════════════════════
@dataclass
class StationParam:
    """储能型充电站物理参数"""
    station_id: int
    # 电网接入
    grid_limit_kw: float = 560.0
    # 站内储能
    bess_energy_kwh: float = 225.0
    bess_power_kw: float = 800.0
    soc_min: float = 0.10
    soc_max: float = 0.90
    soc0: float = 0.50
    eta_c: float = 0.95             # 标称充电效率
    eta_d: float = 0.95             # 标称放电效率
    # 充电服务
    fast_max_kw: float = 1000.0     # 快充桩总额定功率
    slow_max_kw: float = 7.0        # 单桩慢充功率


@dataclass
class MarketParam:
    """电力市场参数"""
    # 充电服务价格
    fast_price: float = 1.35        # 快充电价(元/kWh)
    slow_price: float = 0.80        # 慢充电价(元/kWh)
    # 中长期合约市场
    contract_price: float = 420.0   # 合约价格(元/MWh)
    contract_penalty: float = 1.00  # 合约偏差罚金(元/kWh)
    contract_min_kw: float = 500.0  # 合约最低容量门槛(kW)
    # 现货市场
    spot_discount: float = 20.0     # 现货折价(元/MWh)
    spot_penalty: float = 0.15      # 现货偏差罚金(元/kWh)
    # 需求响应
    dr_price: float = 600.0         # 需求响应补偿(元/MWh)
    dr_top_k: int = 12              # 需求响应时段数
    # 辅助服务(调峰)
    as_price: float = 120.0         # 辅助服务价格(元/MWh)
    as_penalty: float = 1.00        # 辅助服务偏差罚金(元/kWh)
    as_min_kw: float = 1000.0       # 辅助服务最低容量(kW)
    as_min_dur: float = 1.0         # 辅助服务最低持续时间(h)
    # 可靠性
    reliability: float = 0.80
    # 时段掩码
    reserve_mask: np.ndarray = field(default=None, repr=False)
    contract_mask: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.reserve_mask is None:
            self.reserve_mask = ((HOURS >= 17) & (HOURS < 21)).astype(float)
        if self.contract_mask is None:
            self.contract_mask = ((HOURS >= 18) & (HOURS < 21)).astype(float)


@dataclass
class ThermalParam:
    """电热耦合参数"""
    # 热储能
    heat_max_kwh: float = 60.0      # 热储能容量(kWh)
    heat_init_kwh: float = 0.0      # 初始热量
    heat_loss: float = 0.02         # 每时段热损失率
    heat_charge_eff: float = 0.95   # 主动储热效率
    heat_charge_max_kw: float = 120.0  # 主动储热最大功率(kW)
    # 电加热
    heat_elec_eff: float = 0.80     # 电加热效率(电→热)
    # 温度对储能的影响
    temp_power_ref_c: float = 20.0  # 参考温度(℃)
    temp_power_slope_per_c: float = 0.01  # 每℃功率折减率
    temp_power_floor: float = 0.70  # 最低功率因子
    # 热管理对市场能力的影响
    thermal_bess_reserve_ratio: float = 0.00
    heat_value_factor: float = 1.00
    enable_market_thermal_derate: bool = False
    heat_market_release_ratio: float = 0.45
    market_uplift_cap_kw: float = 900.0
    # 热套利信号系数
    h_elec_penalty_factor: float = 0.35
    h_ch_incentive_factor: float = 0.18
    heat_supply_revenue_factor: float = 0.40
    low_price_quantile: float = 0.35


# ════════════════════════════════════════════════════════════════
# 电价/温度数据
# ════════════════════════════════════════════════════════════════
def load_price_data():
    """从guangdong.xlsx加载电价，选取冬/夏典型日"""
    df = pd.read_excel(EXCEL_FILE)
    df["DATE"] = pd.to_datetime(df["DATE"])
    df["day"] = df["DATE"].dt.date
    df["month"] = df["DATE"].dt.month
    daily = df.groupby(["month", "day"])["日前节点电价(元/MWh)"].agg(["min", "max"])
    daily["spread"] = daily["max"] - daily["min"]
    wd = daily.loc[1, "spread"].idxmax()   # 冬季最大价差日
    sd = daily.loc[8, "spread"].idxmax()   # 夏季最大价差日
    pw = df[df["day"] == wd]["日前节点电价(元/MWh)"].to_numpy()
    ps = df[df["day"] == sd]["日前节点电价(元/MWh)"].to_numpy()
    if len(pw) != N or len(ps) != N:
        raise ValueError("电价数据长度不是96")
    return wd, sd, pw, ps


def daily_temp_curve(t_min, t_max):
    """日温度正弦曲线"""
    c, a = (t_min + t_max) / 2, (t_max - t_min) / 2
    return c + a * np.sin(2 * np.pi * (HOURS - 9) / 24)


# ════════════════════════════════════════════════════════════════
# 电热耦合物理函数
# ════════════════════════════════════════════════════════════════
def kappa_winter(T):
    """冬季热管理需求系数: 温度越低需求越大"""
    return np.clip((20 - T) / 12, 0, 1)

def kappa_summer(T):
    """夏季热管理需求系数: 温度越高需求越大"""
    return np.clip((T - 26) / 8, 0, 1)

def eta_from_temp(T, eta_nom=0.95):
    """温度→储能充放电效率折减"""
    return np.clip(eta_nom - np.maximum(20 - T, 0) * 0.003, 0.80, eta_nom)

def soc_range_from_temp(T, soc_min0=0.10, soc_max0=0.90):
    """温度→SOC安全范围收窄"""
    soc_min = np.clip(soc_min0 + np.maximum(10 - T, 0) * 0.01, soc_min0, 0.25)
    soc_max = np.clip(soc_max0 - np.maximum(10 - T, 0) * 0.005, 0.80, soc_max0)
    return soc_min, soc_max

def power_factor_from_temp(T, ref_c=20.0, slope_per_c=0.01, floor=0.70):
    """温度→储能功率折减因子"""
    return np.clip(1.0 - np.maximum(ref_c - T, 0.0) * slope_per_c, floor, 1.0)


# ════════════════════════════════════════════════════════════════
# 充电需求生成
# ════════════════════════════════════════════════════════════════
def build_fast_demand(w):
    """生成快充需求曲线(确定性)"""
    p = np.zeros(N)
    for h in FAST_SVC_HOURS:
        p += 1000 * np.exp(-0.5 * ((HOURS - h) / 0.5) ** 2)
    return np.minimum(w * 0.32 * p, 1000.0)

def build_slow_demand(w, seed):
    """生成单条慢充需求样本(随机)"""
    rng = np.random.default_rng(seed)
    p = np.zeros(N)
    for h in range(24):
        for _ in range(rng.poisson(w * SLOW_CAR_MEAN[h])):
            e = max(rng.normal(25, 8), 5)
            stay = rng.uniform(2, 8)
            st = h * 4 + int(rng.integers(0, 4))
            dur = max(1, int(stay / DT))
            p[st:min(st + dur, N)] += min(7.0, e / max(stay, DT))
    return p

def build_slow_samples(w, base_seed, n_samples=N_SLOW_SAMPLES):
    """生成慢充蒙特卡洛样本集"""
    return np.array([build_slow_demand(w, base_seed + 1000 * s)
                     for s in range(n_samples)], dtype=float)


# ════════════════════════════════════════════════════════════════
# 场景缩减 (K-means)
# ════════════════════════════════════════════════════════════════
def _kmeans_pp_init(samples, k, rng):
    n = samples.shape[0]
    centers = np.empty((k, samples.shape[1]))
    centers[0] = samples[int(rng.integers(0, n))]
    d2 = np.sum((samples - centers[0]) ** 2, axis=1)
    for j in range(1, k):
        total = d2.sum()
        if total <= 1e-12:
            centers[j] = samples[int(rng.integers(0, n))]
        else:
            idx = int(rng.choice(n, p=d2 / total))
            centers[j] = samples[idx]
            d2 = np.minimum(d2, np.sum((samples - centers[j]) ** 2, axis=1))
    return centers

def reduce_scenarios(samples, n_rep=N_SLOW_SCENARIOS, seed=BASE_SEED, max_iter=50):
    """K-means场景缩减，返回代表场景和概率"""
    samples = np.asarray(samples, dtype=float)
    n = samples.shape[0]
    n_rep = int(min(max(1, n_rep), n))
    rng = np.random.default_rng(seed)
    centers = _kmeans_pp_init(samples, n_rep, rng)
    labels = np.full(n, -1, dtype=int)
    for _ in range(max_iter):
        dist = ((samples[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = dist.argmin(axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_rep):
            members = samples[labels == k]
            centers[k] = members.mean(axis=0) if len(members) else samples[int(rng.integers(0, n))]
    rep = np.zeros_like(centers)
    probs = np.zeros(n_rep)
    for k in range(n_rep):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            rep[k] = centers[k]
            continue
        d = ((samples[idx] - centers[k]) ** 2).sum(axis=1)
        rep[k] = samples[idx[d.argmin()]]
        probs[k] = len(idx) / n
    mask = probs > 1e-12
    rep = np.maximum(rep[mask], 0.0)
    probs = probs[mask]
    order = np.argsort(-probs)
    return rep[order], probs[order] / probs[order].sum()

def build_station_data():
    """生成所有站点的快充/慢充需求数据"""
    fast = np.array([build_fast_demand(TRAFFIC_WEIGHTS[i]) for i in range(N_STATIONS)])
    slow, slow_probs = [], []
    for i in range(N_STATIONS):
        samples = build_slow_samples(TRAFFIC_WEIGHTS[i], BASE_SEED + 10000 * i)
        rp, pr = reduce_scenarios(samples, N_SLOW_SCENARIOS, BASE_SEED + 77 * i)
        slow.append(rp)
        slow_probs.append(pr)
    return fast, np.array(slow, dtype=float), np.array(slow_probs, dtype=float)


# ════════════════════════════════════════════════════════════════
# 市场准入评估
# ════════════════════════════════════════════════════════════════
def assess_eligibility(stations, mp):
    """评估VPP聚合后的市场准入资格"""
    P_b = sum(s.bess_power_kw for s in stations)
    P_g = sum(s.grid_limit_kw for s in stations)
    E_u = sum((s.soc_max - s.soc_min) * s.bess_energy_kwh for s in stations)
    P_r = mp.reliability * min(P_b, P_g)
    T_d = E_u / P_r if P_r > 1e-6 else 0.0
    roc = min(P_r, E_u / mp.as_min_dur) if mp.as_min_dur > 1e-9 else P_r
    can_ps = roc >= mp.as_min_kw - 1e-9
    return {
        "P_reliable_kW": P_r,
        "E_usable_kWh": E_u,
        "T_duration_h": T_d,
        "reserve_offer_cap_kW": roc,
        "can_mid_long": P_r >= mp.contract_min_kw,
        "can_spot": True,
        "can_demand_response": True,
        "can_peak_shaving": can_ps,
    }

def build_dr_mask(price, top_k):
    """需求响应时段掩码(电价最高的top_k个时段)"""
    mask = np.zeros(N)
    mask[np.argsort(price)[-top_k:]] = 1.0
    return mask


# ════════════════════════════════════════════════════════════════
# 绘图样式
# ════════════════════════════════════════════════════════════════
def _pick_font(candidates):
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

def setup_plot_style():
    zh = _pick_font(["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                     "Source Han Sans CN", "PingFang SC", "Arial Unicode MS"])
    en = _pick_font(["Times New Roman", "Times", "DejaVu Serif"])
    family = []
    if zh: family.append(zh)
    if en and en not in family: family.append(en)
    if "DejaVu Sans" not in family: family.append("DejaVu Sans")
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": family,
        "axes.unicode_minus": False, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10.5,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "axes.linewidth": 0.9,
        "grid.linewidth": 0.7, "lines.linewidth": 1.8,
        "mathtext.fontset": "stix",
    })