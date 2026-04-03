# -*- coding: utf-8 -*-
"""
真正双层模型 — 上层VPP模型 + KKT嵌入求解

上层决策变量（VPP聚合商）：
  - 激励价格 π_c[t], π_s[t], π_r[t]
  - 各站分配量 alloc_c[i][t], alloc_s[i][t], alloc_r[i][t]
  - 市场申报策略 contract_ratio, spot_ratio, reserve_kw

上层目标：max VPP利润 = 市场收入 - 激励成本
  即 min -(市场收入) + 激励成本

下层变量通过KKT最优性条件锁定为最优响应，而非上层直接优化。

求解器：Gurobi MIQCP (NonConvex=2)
  - 驻点条件中 π*DT 项与对偶变量的交互产生双线性
  - 上层目标中 π*g 产生双线性
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from model.vpp_cs_bilevel_base import (
    N, DT, HOURS, BIG_M, THROUGHPUT_COST, FAIRNESS_CS_PENALTY,
    power_factor_from_temp,
)
from model.vpp_cs_bilevel_lower import build_lower_kkt


def solve_bilevel(price_mwh, kappa, temp, alpha, beta,
                  stations, station_fast, station_slow, station_slow_probs,
                  eligibility, mp, tp, dr_mask, n_st=None):
    """构建并求解真正的双层协调优化模型（KKT嵌入）。"""
    if n_st is None:
        n_st = len(stations)

    # ── 市场价格信号 ──
    P_rel = float(eligibility["P_reliable_kW"])
    sell_p = np.maximum(price_mwh - mp.spot_discount, 0) / 1000
    buy_p = price_mwh / 1000
    high_mask = (price_mwh >= np.percentile(price_mwh, 75)).astype(float)

    cup = np.full(N, mp.contract_price / 1000.0)
    sup = sell_p.copy()
    rup = np.full(N, mp.as_price / 1000.0 * mp.reliability)

    # ── 市场可用功率 ──
    tpf = power_factor_from_temp(temp, tp.temp_power_ref_c,
                                 tp.temp_power_slope_per_c, tp.temp_power_floor)
    agg_fast = station_fast[:n_st].sum(axis=0)
    Prel_prof = np.minimum(P_rel, tpf * P_rel)
    tmu = tp.heat_market_release_ratio * np.minimum(
        alpha * kappa * agg_fast, tp.market_uplift_cap_kw)
    Pmkt = np.maximum(Prel_prof + tmu, 0.0)

    # ── 慢充期望需求（确定性化） ──
    slow_expected = []
    for i in range(n_st):
        probs = station_slow_probs[i]
        probs = probs / probs.sum()
        exp_slow = np.tensordot(probs, station_slow[i], axes=(0, 0))
        slow_expected.append(exp_slow)

    # ═══════════════════════════════════════════
    # Gurobi 模型
    # ═══════════════════════════════════════════
    mdl = gp.Model("VPP_CS_Bilevel_KKT")
    mdl.Params.OutputFlag = 1
    mdl.Params.TimeLimit = 1200
    mdl.Params.MIPGap = 0.03
    mdl.Params.NonConvex = 2
    mdl.Params.MIPFocus = 1
    mdl.Params.Threads = 0

    # ═══════════════════════════════════════════
    # 上层变量：VPP市场策略
    # ═══════════════════════════════════════════
    contract_ratio = mdl.addVar(lb=0, ub=1.0 if eligibility["can_mid_long"] else 0.0,
                                name="contract_ratio")
    spot_ratio = mdl.addVar(lb=0, ub=1.0 if eligibility["can_spot"] else 0.0,
                            name="spot_ratio")

    rpc = (float(np.min(Pmkt[mp.reserve_mask > 0.5]))
           if np.any(mp.reserve_mask > 0.5) else float(np.min(Pmkt)))
    roc = min(rpc, float(eligibility.get("reserve_offer_cap_kW", 0.0)))
    reserve_kw = mdl.addVar(lb=0, ub=roc if eligibility["can_peak_shaving"] else 0.0,
                            name="reserve_kw")
    reserve_on = mdl.addVar(vtype=GRB.BINARY, name="reserve_on")
    if not eligibility["can_peak_shaving"]:
        reserve_on.ub = 0

    # 市场申报曲线
    cc = mdl.addVars(N, lb=0, name="cc")
    so = mdl.addVars(N, lb=0, name="so")
    rt = mdl.addVars(N, lb=0, name="rt")
    for t in range(N):
        cc[t].ub = float(Pmkt[t])
        so[t].ub = float(Pmkt[t])
        rt[t].ub = float(Pmkt[t])

    # 激励价格（上层核心决策变量）
    pi_c = mdl.addVars(N, lb=0, name="pi_c")
    pi_s = mdl.addVars(N, lb=0, name="pi_s")
    pi_r = mdl.addVars(N, lb=0, name="pi_r")
    for t in range(N):
        pi_c[t].ub = float(cup[t])
        pi_s[t].ub = float(sup[t])
        pi_r[t].ub = float(rup[t])

    # 分配变量（上层核心决策变量）
    alloc_c = [[mdl.addVar(lb=0, ub=float(Pmkt[t]), name=f"ac_{i}_{t}")
                for t in range(N)] for i in range(n_st)]
    alloc_s = [[mdl.addVar(lb=0, ub=float(Pmkt[t]), name=f"as_{i}_{t}")
                for t in range(N)] for i in range(n_st)]
    alloc_r = [[mdl.addVar(lb=0, ub=float(Pmkt[t]), name=f"ar_{i}_{t}")
                for t in range(N)] for i in range(n_st)]

    # ═══════════════════════════════════════════
    # 下层：各站KKT嵌入
    # ═══════════════════════════════════════════
    st_models = []
    for i in range(n_st):
        sm = build_lower_kkt(
            mdl, stations[i], price_mwh, station_fast[i],
            slow_expected[i], kappa, temp, alpha, beta, mp, tp,
            pi_c, pi_s, pi_r,
            alloc_c[i], alloc_s[i], alloc_r[i],
            station_idx=i)
        st_models.append(sm)

    mdl.update()

    # ═══════════════════════════════════════════
    # 上层约束：市场申报
    # ═══════════════════════════════════════════
    for t in range(N):
        mdl.addConstr(cc[t] == contract_ratio * float(Pmkt[t] * mp.contract_mask[t]),
                      name=f"cc_def_{t}")
        mdl.addConstr(rt[t] == reserve_kw * float(mp.reserve_mask[t]),
                      name=f"rt_def_{t}")
        # 分配求和 = 总申报
        mdl.addConstr(gp.quicksum(alloc_c[i][t] for i in range(n_st)) == cc[t],
                      name=f"ac_sum_{t}")
        mdl.addConstr(gp.quicksum(alloc_s[i][t] for i in range(n_st)) == so[t],
                      name=f"as_sum_{t}")
        mdl.addConstr(gp.quicksum(alloc_r[i][t] for i in range(n_st)) == rt[t],
                      name=f"ar_sum_{t}")

    # 市场容量约束
    for t in range(N):
        mdl.addConstr(so[t] <= spot_ratio * float(Pmkt[t] * high_mask[t]),
                      name=f"so_ub_{t}")
        mdl.addConstr(cc[t] + rt[t] <= float(Pmkt[t]), name=f"cap_cr_{t}")
        mdl.addConstr(cc[t] + rt[t] + so[t] <= float(Pmkt[t]), name=f"cap_all_{t}")

    # 备用约束
    mdl.addConstr(reserve_kw <= roc * reserve_on, name="rk_on")
    mdl.addConstr(reserve_kw >= mp.as_min_kw * reserve_on, name="rk_min")

    # ═══════════════════════════════════════════
    # IR约束（站端利润≥0，额外保障）
    # ═══════════════════════════════════════════
    for i in range(n_st):
        sm = st_models[i]
        profit = gp.QuadExpr()
        for t in range(N):
            profit += mp.fast_price * DT * sm["p_fast"][t]
            profit += mp.slow_price * DT * sm["p_slow"][t]
            profit += DT * pi_c[t] * sm["g_contract"][t]
            profit += DT * pi_s[t] * sm["g_spot"][t]
            profit += DT * pi_r[t] * sm["reserve"][t]
            profit -= float(buy_p[t]) * DT * sm["g_buy"][t]
            profit -= THROUGHPUT_COST * DT * sm["ch"][t]
            profit -= THROUGHPUT_COST * DT * sm["dis"][t]
            profit -= float(buy_p[t]) * DT * sm["h_elec"][t]
            profit -= float(buy_p[t]) * DT * sm["h_ch_grid"][t]
            profit -= mp.contract_penalty * DT * sm["cs"][t]
            profit -= mp.spot_penalty * DT * sm["ss"][t]
            profit -= mp.as_penalty * DT * sm["rs"][t]
        mdl.addQConstr(profit >= 0, name=f"IR_{i}")

    # ═══════════════════════════════════════════
    # 上层目标：min -(市场收入) + 激励成本
    # VPP利润 = Σ(市场价×下层交付) - Σ(π×下层交付)
    # ═══════════════════════════════════════════
    obj = gp.QuadExpr()

    for i in range(n_st):
        sm = st_models[i]
        for t in range(N):
            # -市场收入 (VPP按市场价卖电)
            obj -= float(cup[t]) * DT * sm["g_contract"][t]
            obj -= float(sup[t]) * DT * sm["g_spot"][t]
            obj -= float(rup[t]) * DT * sm["reserve"][t]
            # +激励成本 (VPP支付给站, 双线性: π×g)
            obj += DT * pi_c[t] * sm["g_contract"][t]
            obj += DT * pi_s[t] * sm["g_spot"][t]
            obj += DT * pi_r[t] * sm["reserve"][t]

    mdl.setObjective(obj, GRB.MINIMIZE)

    # ═══════════════════════════════════════════
    # 求解
    # ═══════════════════════════════════════════
    mdl.update()
    print(f"    真正双层KKT嵌入: {mdl.NumVars}vars / {mdl.NumBinVars}bin / "
          f"{mdl.NumConstrs}constr / {mdl.NumQConstrs}qconstr")

    mdl.optimize()

    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT) or mdl.SolCount == 0:
        print(f"    Gurobi status: {mdl.Status}, 无可行解")
        return None

    status_msg = {GRB.OPTIMAL: "Optimal", GRB.SUBOPTIMAL: "Suboptimal",
                  GRB.TIME_LIMIT: "Time limit"}.get(mdl.Status, f"Status={mdl.Status}")

    # ═══════════════════════════════════════════
    # 提取结果
    # ═══════════════════════════════════════════
    pic_val = np.array([pi_c[t].X for t in range(N)])
    pis_val = np.array([pi_s[t].X for t in range(N)])
    pir_val = np.array([pi_r[t].X for t in range(N)])

    st_results = []
    for i in range(n_st):
        sm = st_models[i]

        def _v(vd):
            return np.array([vd[t].X for t in range(N)])

        g_buy_v = _v(sm["g_buy"])
        p_fast_v = _v(sm["p_fast"])
        p_slow_v = _v(sm["p_slow"])
        g_contract_v = _v(sm["g_contract"])
        g_spot_v = _v(sm["g_spot"])
        reserve_v = _v(sm["reserve"])
        ch_v = _v(sm["ch"])
        dis_v = _v(sm["dis"])
        e_v = _v(sm["e"])
        h_v = _v(sm["h"])
        h_elec_v = _v(sm["h_elec"])
        h_ch_grid_v = _v(sm["h_ch_grid"])
        q_heat_v = _v(sm["q_heat"])
        cs_v = _v(sm["cs"])
        ss_v = _v(sm["ss"])
        rs_v = _v(sm["rs"])

        user_rev = float(mp.fast_price * np.sum(p_fast_v) * DT
                         + mp.slow_price * np.sum(p_slow_v) * DT)
        buy_cost = float(np.sum(buy_p * g_buy_v * DT))
        tc = float(np.sum((ch_v + dis_v) * THROUGHPUT_COST * DT))
        h_elec_cost = float(np.sum(buy_p * h_elec_v * DT))
        h_ch_grid_cost = float(np.sum(buy_p * h_ch_grid_v * DT))
        heat_offset = float(np.sum(q_heat_v * buy_p * DT
                                   / max(tp.heat_elec_eff, 1e-9)) * tp.heat_value_factor)

        inc_c = float(np.sum(pic_val * g_contract_v * DT))
        inc_s = float(np.sum(pis_val * g_spot_v * DT))
        inc_r = float(np.sum(pir_val * reserve_v * DT))
        inc_total = inc_c + inc_s + inc_r

        cs_kwh = float(np.sum(cs_v) * DT)
        ss_kwh = float(np.sum(ss_v) * DT)
        rs_kwh = float(np.sum(rs_v) * DT)
        penalty_cost = (mp.contract_penalty * cs_kwh + mp.spot_penalty * ss_kwh
                        + mp.as_penalty * rs_kwh)
        thermal_cost = h_elec_cost + h_ch_grid_cost
        station_profit = user_rev + inc_total - buy_cost - tc - penalty_cost - thermal_cost

        st_results.append({
            "station_id": i + 1,
            "g_buy": g_buy_v, "p_fast": p_fast_v, "p_slow_mean": p_slow_v,
            "g_contract": g_contract_v, "g_spot": g_spot_v, "reserve": reserve_v,
            "ch": ch_v, "dis": dis_v, "e": e_v, "h": h_v,
            "h_elec": h_elec_v, "h_ch_grid": h_ch_grid_v, "q_heat": q_heat_v,
            "cs": cs_v, "ss": ss_v, "rs": rs_v,
            "alloc_contract": np.array([alloc_c[i][t].X for t in range(N)]),
            "alloc_spot": np.array([alloc_s[i][t].X for t in range(N)]),
            "alloc_reserve": np.array([alloc_r[i][t].X for t in range(N)]),
            "contract_del": float(np.sum(g_contract_v) * DT),
            "spot_del": float(np.sum(g_spot_v) * DT),
            "reserve_com": float(np.sum(reserve_v * mp.reserve_mask) * DT),
            "cs_kwh": cs_kwh, "ss_kwh": ss_kwh, "rs_kwh": rs_kwh,
            "buy_cost": buy_cost, "user_rev": user_rev, "tc": tc,
            "heat_offset": heat_offset,
            "heat_supply_kwh": float(np.sum(q_heat_v) * DT),
            "h_elec_cost": h_elec_cost, "h_ch_grid_cost": h_ch_grid_cost,
            "fast_ratio": float(np.sum(p_fast_v) / max(np.sum(station_fast[i]), 1e-9)),
            "slow_ratio": float(np.sum(p_slow_v) / max(np.sum(slow_expected[i]), 1e-9)),
            "eta_c_t": sm["eta_c_t"], "eta_d_t": sm["eta_d_t"],
            "p_factor_t": sm["p_factor_t"],
            "incentive_rev_contract": inc_c,
            "incentive_rev_spot": inc_s,
            "incentive_rev_reserve": inc_r,
            "incentive_total": inc_total,
            "penalty_cost": penalty_cost,
            "thermal_cost": thermal_cost,
            "station_profit": station_profit,
        })

    # ═══════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════
    agg_contract = sum(r["contract_del"] for r in st_results)
    agg_spot = sum(r["spot_del"] for r in st_results)
    agg_reserve = sum(r["reserve_com"] for r in st_results)
    agg_cs = sum(r["cs_kwh"] for r in st_results)
    agg_ss = sum(r["ss_kwh"] for r in st_results)
    agg_rs = sum(r["rs_kwh"] for r in st_results)
    total_dis_dr = float(sum(np.sum(r["dis"] * dr_mask) * DT for r in st_results))

    contract_rev = agg_contract * mp.contract_price / 1000
    spot_rev = float(sum(np.sum(r["g_spot"] * sell_p * DT) for r in st_results))
    reserve_rev = agg_reserve * mp.as_price / 1000
    dr_rev = total_dis_dr * mp.dr_price / 1000
    penalty = agg_cs * mp.contract_penalty + agg_ss * mp.spot_penalty + agg_rs * mp.as_penalty
    total_incentive = sum(r["incentive_total"] for r in st_results)

    vpp_profit = contract_rev + spot_rev + reserve_rev + dr_rev - total_incentive
    stations_profit = sum(r["station_profit"] for r in st_results)
    system_profit = vpp_profit + stations_profit

    return {
        "milp_status": status_msg,
        "vpp_profit": vpp_profit,
        "stations_total_profit": stations_profit,
        "system_profit": system_profit,
        "user_rev": sum(r["user_rev"] for r in st_results),
        "contract_rev": contract_rev,
        "spot_rev": spot_rev,
        "reserve_rev": reserve_rev,
        "dr_rev": dr_rev,
        "buy_cost": sum(r["buy_cost"] for r in st_results),
        "throughput_cost": sum(r["tc"] for r in st_results),
        "penalty": penalty,
        "incentive_cost": total_incentive,
        "heat_offset_value": sum(r["heat_offset"] for r in st_results),
        "h_elec_cost": sum(r["h_elec_cost"] for r in st_results),
        "h_ch_grid_cost": sum(r["h_ch_grid_cost"] for r in st_results),
        "thermal_net_benefit": (sum(r["heat_offset"] for r in st_results)
                                - sum(r["h_elec_cost"] for r in st_results)
                                - sum(r["h_ch_grid_cost"] for r in st_results)),
        "heat_offset_kwh": float(sum(r["heat_supply_kwh"] for r in st_results)),
        "contract_ratio": float(contract_ratio.X),
        "spot_ratio": float(spot_ratio.X),
        "reserve_kw": float(reserve_kw.X),
        "pi_contract": pic_val,
        "pi_spot": pis_val,
        "pi_reserve": pir_val,
        "contract_curve": np.array([cc[t].X for t in range(N)]),
        "spot_offer": np.array([so[t].X for t in range(N)]),
        "reserve_target": np.array([rt[t].X for t in range(N)]),
        "market_power_profile": Pmkt,
        "temp_power_factor": tpf,
        "station_results": st_results,
        "n_total_vars": mdl.NumVars,
        "n_binary_vars": mdl.NumBinVars,
        "n_qconstr": mdl.NumQConstrs,
        "upper_objective": float(mdl.ObjVal),
    }