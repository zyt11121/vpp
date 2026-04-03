# -*- coding: utf-8 -*-
"""
终极版双层模型 — 下层充电站模型
储能型充电站综合运营单元：充电服务 + 储能调度 + 电热耦合

直接用 Gurobi 变量建模，返回变量引用供上层嵌入。
不再走 scipy sparse → KKT 矩阵路线。

下层决策变量：
  g_buy[t]      — 从电网购电功率(kW)
  p_fast[t]     — 快充服务功率(kW)
  p_slow[s,t]   — 慢充服务功率(kW), s=场景
  g_contract[t] — 合约市场交付功率(kW)
  g_spot[t]     — 现货市场交付功率(kW)
  reserve[t]    — 辅助服务备用容量(kW)
  ch[t], dis[t] — 储能充/放电功率(kW)
  e[t]          — 储能SOC能量(kWh)
  h[t]          — 热储能能量(kWh)
  h_elec[t]     — 电加热功率(kW)
  h_ch_grid[t]  — 主动储热功率(kW)
  q_heat[t]     — 热储能供热功率(kW)
  cs[t],ss[t],rs[t] — 合约/现货/备用偏差松弛(kW)

下层目标（充电站视角）：
  min  购电成本 + 吞吐成本 + 偏差罚金 - 充电服务收入
  注意：g_contract/g_spot/reserve 的目标系数=0
        激励价格 π 由上层通过 KKT stationarity 注入
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from vpp_cs_bilevel_base import (
    N, DT, HOURS, BIG_M, THROUGHPUT_COST,
    eta_from_temp, soc_range_from_temp, power_factor_from_temp,
)


def build_station_model(mdl, station, price_mwh, fast_kw,
                        slow_scenarios, scenario_probs,
                        kappa, temp, alpha, beta, mp, tp,
                        station_idx):
    """
    在 Gurobi 模型 mdl 中为第 station_idx 个充电站添加所有变量和约束。

    返回字典包含：
      - 所有 Gurobi 变量引用
      - 温度相关物理参数（供上层使用）
      - KKT 所需的 stationarity 行信息
    """
    S = slow_scenarios.shape[0]
    scenario_probs = np.asarray(scenario_probs, dtype=float)
    scenario_probs = scenario_probs / scenario_probs.sum()
    bp = price_mwh / 1000.0  # 元/kWh
    sid = station_idx  # 简写

    # ── 温度相关物理参数 ──
    eta_c_t = eta_from_temp(temp, station.eta_c)
    eta_d_t = eta_from_temp(temp, station.eta_d)
    soc_min_t, soc_max_t = soc_range_from_temp(temp, station.soc_min, station.soc_max)
    p_factor_t = power_factor_from_temp(temp, ref_c=tp.temp_power_ref_c,
                                        slope_per_c=tp.temp_power_slope_per_c,
                                        floor=tp.temp_power_floor)

    # ════════════════════════════════════════════
    # 决策变量
    # ════════════════════════════════════════════
    g_buy = mdl.addVars(N, lb=0, ub=station.grid_limit_kw, name=f"s{sid}_gbuy")
    p_fast = mdl.addVars(N, lb=0, name=f"s{sid}_pfast")
    g_contract = mdl.addVars(N, lb=0, ub=station.grid_limit_kw, name=f"s{sid}_gc")
    g_spot = mdl.addVars(N, lb=0, ub=station.grid_limit_kw, name=f"s{sid}_gs")
    reserve = mdl.addVars(N, lb=0, name=f"s{sid}_rv")
    ch = mdl.addVars(N, lb=0, name=f"s{sid}_ch")
    dis = mdl.addVars(N, lb=0, name=f"s{sid}_dis")
    e = mdl.addVars(N, lb=0, name=f"s{sid}_e")
    h = mdl.addVars(N, lb=0, ub=tp.heat_max_kwh, name=f"s{sid}_h")
    h_elec = mdl.addVars(N, lb=0, name=f"s{sid}_helec")
    h_ch_grid = mdl.addVars(N, lb=0, ub=tp.heat_charge_max_kw, name=f"s{sid}_hchg")
    q_heat = mdl.addVars(N, lb=0, ub=tp.heat_max_kwh / DT, name=f"s{sid}_qheat")
    cs = mdl.addVars(N, lb=0, name=f"s{sid}_cs")
    ss = mdl.addVars(N, lb=0, name=f"s{sid}_ss")
    rs = mdl.addVars(N, lb=0, name=f"s{sid}_rs")
    p_slow = [[mdl.addVar(lb=0, ub=float(slow_scenarios[s, t]),
                           name=f"s{sid}_pslow_{s}_{t}")
               for t in range(N)] for s in range(S)]

    # 设置时变上界
    for t in range(N):
        p_fast[t].ub = float(fast_kw[t])
        reserve[t].ub = float(station.bess_power_kw * p_factor_t[t])
        ch[t].ub = float(station.bess_power_kw * p_factor_t[t])
        dis[t].ub = float(station.bess_power_kw * p_factor_t[t])
        e[t].lb = float(soc_min_t[t] * station.bess_energy_kwh)
        e[t].ub = float(soc_max_t[t] * station.bess_energy_kwh)

    # ════════════════════════════════════════════
    # 约束
    # ════════════════════════════════════════════

    # ── 模块1: 充电服务约束 ──
    # 快充功率 <= 需求上限 (已在变量ub中)
    # 慢充功率 <= 场景需求 (已在变量ub中)

    # ── 模块2: 站内功率平衡 ──
    # 对每个慢充场景: 购电+放电 = 快充+慢充+热管理+储能充电+市场交付
    for s in range(S):
        for t in range(N):
            mdl.addConstr(
                g_buy[t] + dis[t]
                == p_fast[t] + p_slow[s][t] + h_elec[t] + h_ch_grid[t]
                   + ch[t] + g_contract[t] + g_spot[t],
                name=f"s{sid}_bal_{s}_{t}")

    # ── 模块3: 市场交付与偏差 ──
    # 合约: g_contract[t] + cs[t] = alloc_contract[t] (alloc由上层注入)
    # 现货: g_spot[t] + ss[t] = alloc_spot[t]
    # 备用: reserve[t] + rs[t] = alloc_reserve[t]
    # 这些等式的右端由上层分配变量决定，此处先建立结构
    constr_contract = {}
    constr_spot = {}
    constr_reserve = {}
    for t in range(N):
        constr_contract[t] = mdl.addConstr(
            g_contract[t] + cs[t] == 0, name=f"s{sid}_cdel_{t}")
        constr_spot[t] = mdl.addConstr(
            g_spot[t] + ss[t] == 0, name=f"s{sid}_sdel_{t}")
        constr_reserve[t] = mdl.addConstr(
            reserve[t] + rs[t] == 0, name=f"s{sid}_rdel_{t}")

    # ── 模块4: 储能运行约束 ──
    # SOC演化
    for t in range(N):
        if t == 0:
            mdl.addConstr(
                e[t] == station.soc0 * station.bess_energy_kwh
                        + eta_c_t[t] * DT * ch[t] - DT / eta_d_t[t] * dis[t],
                name=f"s{sid}_soc_{t}")
        else:
            mdl.addConstr(
                e[t] == e[t-1] + eta_c_t[t] * DT * ch[t] - DT / eta_d_t[t] * dis[t],
                name=f"s{sid}_soc_{t}")
    # 终端SOC回归
    mdl.addConstr(e[N-1] == station.soc0 * station.bess_energy_kwh,
                  name=f"s{sid}_soc_end")

    # 电网容量约束
    for t in range(N):
        mdl.addConstr(g_buy[t] + g_contract[t] + g_spot[t] <= station.grid_limit_kw,
                      name=f"s{sid}_grid_{t}")

    # 市场交付不超过可用放电+购电
    for t in range(N):
        mdl.addConstr(g_contract[t] + g_spot[t] + reserve[t] <= dis[t] + g_buy[t],
                      name=f"s{sid}_mkt_cap_{t}")

    # 备用容量需要SOC支撑
    for t in range(N):
        mdl.addConstr(
            reserve[t] * DT / eta_d_t[t] <= e[t] - soc_min_t[t] * station.bess_energy_kwh,
            name=f"s{sid}_rv_soc_{t}")

    # 放电+备用 <= 功率上限
    for t in range(N):
        mdl.addConstr(dis[t] + reserve[t] <= station.bess_power_kw * p_factor_t[t],
                      name=f"s{sid}_dis_rv_{t}")

    # ── 模块5: 电热耦合约束 ──
    # 热储能动态
    for t in range(N):
        if t == 0:
            mdl.addConstr(
                h[t] == tp.heat_init_kwh
                        + beta * DT * p_fast[t]
                        + tp.heat_charge_eff * DT * h_ch_grid[t]
                        - DT * q_heat[t],
                name=f"s{sid}_heat_{t}")
        else:
            mdl.addConstr(
                h[t] == (1 - tp.heat_loss) * h[t-1]
                        + beta * DT * p_fast[t]
                        + tp.heat_charge_eff * DT * h_ch_grid[t]
                        - DT * q_heat[t],
                name=f"s{sid}_heat_{t}")

    # 热平衡: 热需求 = 热储能供热 + 电加热
    for t in range(N):
        mdl.addConstr(
            q_heat[t] + tp.heat_elec_eff * h_elec[t] == alpha * kappa[t] * p_fast[t],
            name=f"s{sid}_hbal_{t}")

    # ════════════════════════════════════════════
    # 下层目标系数（供KKT stationarity使用）
    # ════════════════════════════════════════════
    # 下层 min: 购电成本 + 吞吐成本 + 偏差罚金 - 充电收入
    # g_contract/g_spot/reserve 系数=0 (激励由上层π注入)
    obj_coeffs = {}
    obj_coeffs["g_buy"] = bp * DT
    obj_coeffs["p_fast"] = -mp.fast_price * DT * np.ones(N)
    obj_coeffs["g_contract"] = np.zeros(N)  # π由上层注入
    obj_coeffs["g_spot"] = np.zeros(N)
    obj_coeffs["reserve"] = np.zeros(N)
    obj_coeffs["ch"] = THROUGHPUT_COST * DT * np.ones(N)
    obj_coeffs["dis"] = THROUGHPUT_COST * DT * np.ones(N)
    obj_coeffs["cs"] = mp.contract_penalty * DT * np.ones(N)
    obj_coeffs["ss"] = mp.spot_penalty * DT * np.ones(N)
    obj_coeffs["rs"] = mp.as_penalty * DT * np.ones(N)
    for s in range(S):
        obj_coeffs[f"p_slow_{s}"] = -scenario_probs[s] * mp.slow_price * DT * np.ones(N)

    return {
        # 变量引用
        "g_buy": g_buy, "p_fast": p_fast,
        "g_contract": g_contract, "g_spot": g_spot, "reserve": reserve,
        "ch": ch, "dis": dis, "e": e,
        "h": h, "h_elec": h_elec, "h_ch_grid": h_ch_grid, "q_heat": q_heat,
        "cs": cs, "ss": ss, "rs": rs,
        "p_slow": p_slow,
        # 约束引用（供上层修改RHS）
        "constr_contract": constr_contract,
        "constr_spot": constr_spot,
        "constr_reserve": constr_reserve,
        # 物理参数
        "eta_c_t": eta_c_t, "eta_d_t": eta_d_t,
        "soc_min_t": soc_min_t, "soc_max_t": soc_max_t,
        "p_factor_t": p_factor_t,
        "scenario_probs": scenario_probs,
        "n_scenarios": S,
        # 目标系数
        "obj_coeffs": obj_coeffs,
    }