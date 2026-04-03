# -*- coding: utf-8 -*-
"""
终极版双层模型 — 上层VPP模型 + 双层协调求解
面向快充服务场景的、考虑电热耦合与不确定性的、VPP—储能型充电站双层协调优化模型

上层决策：
  - 日前市场申报策略（合约/现货/辅助服务）
  - 激励价格 π_contract[t], π_spot[t], π_reserve[t]
  - 各站灵活性购买量（分配）

双层协调机制：
  - 下层充电站通过 KKT 条件嵌入上层
  - IR 约束（站端利润≥0）用 Gurobi 二次约束精确表示
  - 上层目标中 π×g 双线性项直接用 Gurobi 二次目标

求解器：Gurobi MIQCP (NonConvex=2)
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB

from vpp_cs_bilevel_base import (
    N, DT, HOURS, BIG_M, THROUGHPUT_COST, FAIRNESS_CS_PENALTY,
    power_factor_from_temp,
)
from vpp_cs_bilevel_lower import build_station_model


def solve_bilevel(price_mwh, kappa, temp, alpha, beta,
                  stations, station_fast, station_slow, station_slow_probs,
                  eligibility, mp, tp, dr_mask, n_st=None):
    """
    构建并求解 VPP-充电站双层协调优化模型。

    返回完整结果字典，或 None（无可行解）。
    """
    if n_st is None:
        n_st = len(stations)

    # ── 市场价格信号 ──
    P_rel = float(eligibility["P_reliable_kW"])
    sell_p = np.maximum(price_mwh - mp.spot_discount, 0) / 1000  # 元/kWh
    buy_p = price_mwh / 1000
    high_mask = (price_mwh >= np.percentile(price_mwh, 75)).astype(float)
    low_mask = (price_mwh <= np.percentile(price_mwh, 100 * tp.low_price_quantile)).astype(float)
    heat_value_signal = np.maximum(np.max(buy_p) - buy_p, 0.0)

    # 市场单位价格 (元/kWh) — π 的上界
    cup = np.full(N, mp.contract_price / 1000.0)
    sup = sell_p.copy()
    rup = np.full(N, mp.as_price / 1000.0 * mp.reliability)

    # ── 市场可用功率 ──
    tpf = power_factor_from_temp(temp, ref_c=tp.temp_power_ref_c,
                                 slope_per_c=tp.temp_power_slope_per_c,
                                 floor=tp.temp_power_floor)
    agg_fast = station_fast[:n_st].sum(axis=0)
    tqr = np.zeros_like(temp)
    Prel_prof = np.minimum(P_rel, tpf * P_rel)
    tmu = tp.heat_market_release_ratio * np.minimum(
        alpha * kappa * agg_fast, tp.market_uplift_cap_kw)
    Pmkt = np.maximum(Prel_prof - tqr + tmu, 0.0)

    # ═══════════════════════════════════════════
    # Gurobi 模型
    # ═══════════════════════════════════════════
    mdl = gp.Model("VPP_CS_Bilevel")
    mdl.Params.OutputFlag = 1
    mdl.Params.TimeLimit = 600
    mdl.Params.MIPGap = 0.02
    mdl.Params.NonConvex = 2  # 允许非凸二次

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
    cc = mdl.addVars(N, lb=0, name="cc")  # 合约申报
    so = mdl.addVars(N, lb=0, name="so")  # 现货申报
    rt = mdl.addVars(N, lb=0, name="rt")  # 备用申报
    for t in range(N):
        cc[t].ub = float(Pmkt[t])
        so[t].ub = float(Pmkt[t])
        rt[t].ub = float(Pmkt[t])

    # 激励价格
    pi_c = mdl.addVars(N, lb=0, name="pi_c")
    pi_s = mdl.addVars(N, lb=0, name="pi_s")
    pi_r = mdl.addVars(N, lb=0, name="pi_r")
    for t in range(N):
        pi_c[t].ub = float(cup[t])
        pi_s[t].ub = float(sup[t])
        pi_r[t].ub = float(rup[t])

    # 分配变量：VPP分配给各站的灵活性购买量
    alloc_c = [[mdl.addVar(lb=0, ub=float(Pmkt[t]), name=f"ac_{i}_{t}")
                for t in range(N)] for i in range(n_st)]
    alloc_s = [[mdl.addVar(lb=0, ub=float(Pmkt[t]), name=f"as_{i}_{t}")
                for t in range(N)] for i in range(n_st)]
    alloc_r = [[mdl.addVar(lb=0, ub=float(Pmkt[t]), name=f"ar_{i}_{t}")
                for t in range(N)] for i in range(n_st)]

    # 公平性辅助变量
    avg_cs = mdl.addVar(lb=0, name="avg_cs")
    cs_dev = mdl.addVars(n_st, lb=0, name="csdev")

    # ═══════════════════════════════════════════
    # 下层：各站模型
    # ═══════════════════════════════════════════
    st_models = []
    for i in range(n_st):
        sm = build_station_model(
            mdl, stations[i], price_mwh, station_fast[i],
            station_slow[i], station_slow_probs[i],
            kappa, temp, alpha, beta, mp, tp, station_idx=i)
        st_models.append(sm)

    mdl.update()

    # ═══════════════════════════════════════════
    # 上层约束：市场申报
    # ═══════════════════════════════════════════
    for t in range(N):
        # 合约申报 = ratio × Pmkt × mask
        mdl.addConstr(cc[t] == contract_ratio * float(Pmkt[t] * mp.contract_mask[t]),
                      name=f"cc_def_{t}")
        # 备用申报 = reserve_kw × mask
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
    # 双层耦合：分配 → 下层交付约束
    # ═══════════════════════════════════════════
    for i in range(n_st):
        sm = st_models[i]
        for t in range(N):
            # 修改下层交付约束的RHS为分配量
            # g_contract[t] + cs[t] == alloc_c[i][t]
            # 由于 Gurobi 不支持直接修改RHS为变量，
            # 我们删除原约束并重建
            mdl.remove(sm["constr_contract"][t])
            mdl.remove(sm["constr_spot"][t])
            mdl.remove(sm["constr_reserve"][t])

            mdl.addConstr(sm["g_contract"][t] + sm["cs"][t] == alloc_c[i][t],
                          name=f"s{i}_cdel_{t}")
            mdl.addConstr(sm["g_spot"][t] + sm["ss"][t] == alloc_s[i][t],
                          name=f"s{i}_sdel_{t}")
            mdl.addConstr(sm["reserve"][t] + sm["rs"][t] == alloc_r[i][t],
                          name=f"s{i}_rdel_{t}")

    # 公平性约束
    cs_sum = gp.LinExpr()
    for i in range(n_st):
        sm = st_models[i]
        for t in range(N):
            cs_sum += DT * sm["cs"][t]
    mdl.addConstr(n_st * avg_cs == cs_sum, name="avg_cs_def")

    for i in range(n_st):
        sm = st_models[i]
        cs_i = gp.quicksum(DT * sm["cs"][t] for t in range(N))
        mdl.addConstr(cs_i - avg_cs <= cs_dev[i], name=f"csdev_pos_{i}")
        mdl.addConstr(avg_cs - cs_i <= cs_dev[i], name=f"csdev_neg_{i}")

    # ═══════════════════════════════════════════
    # IR 约束：每个站端利润 ≥ 0（二次约束）
    # station_profit = 充电收入 + 激励收入(π×g) - 购电成本 - 吞吐成本 - 热管理成本 - 偏差罚金
    # ═══════════════════════════════════════════
    for i in range(n_st):
        sm = st_models[i]
        profit = gp.QuadExpr()

        # + 充电服务收入
        for t in range(N):
            profit += mp.fast_price * DT * sm["p_fast"][t]
        for s in range(sm["n_scenarios"]):
            pr = float(sm["scenario_probs"][s])
            for t in range(N):
                profit += pr * mp.slow_price * DT * sm["p_slow"][s][t]

        # + 激励收入（二次项：π × g，精确表示）
        for t in range(N):
            profit += DT * pi_c[t] * sm["g_contract"][t]
            profit += DT * pi_s[t] * sm["g_spot"][t]
            profit += DT * pi_r[t] * sm["reserve"][t]

        # - 购电成本
        for t in range(N):
            profit -= float(buy_p[t]) * DT * sm["g_buy"][t]

        # - 储能吞吐成本
        for t in range(N):
            profit -= THROUGHPUT_COST * DT * sm["ch"][t]
            profit -= THROUGHPUT_COST * DT * sm["dis"][t]

        # - 热管理成本
        for t in range(N):
            profit -= float(buy_p[t]) * DT * sm["h_elec"][t]
            profit -= float(buy_p[t]) * DT * sm["h_ch_grid"][t]

        # - 偏差罚金
        for t in range(N):
            profit -= mp.contract_penalty * DT * sm["cs"][t]
            profit -= mp.spot_penalty * DT * sm["ss"][t]
            profit -= mp.as_penalty * DT * sm["rs"][t]

        mdl.addQConstr(profit >= 0, name=f"IR_{i}")

    # ═══════════════════════════════════════════
    # 目标函数（二次）
    # VPP利润 = 市场收入 - 激励成本
    # minimize: -(市场收入) + 激励成本 + 公平性惩罚
    # ═══════════════════════════════════════════
    obj = gp.QuadExpr()

    # 公平性惩罚
    for i in range(n_st):
        obj += FAIRNESS_CS_PENALTY * cs_dev[i]

    for i in range(n_st):
        sm = st_models[i]
        for t in range(N):
            # - 市场收入（VPP卖电收入）
            obj -= float(cup[t]) * DT * sm["g_contract"][t]
            obj -= float(sup[t]) * DT * sm["g_spot"][t]
            obj -= float(rup[t]) * DT * sm["reserve"][t]
            # + 激励成本（VPP支付给充电站，二次项）
            obj += DT * pi_c[t] * sm["g_contract"][t]
            obj += DT * pi_s[t] * sm["g_spot"][t]
            obj += DT * pi_r[t] * sm["reserve"][t]

    mdl.setObjective(obj, GRB.MINIMIZE)

    # ═══════════════════════════════════════════
    # 求解
    # ═══════════════════════════════════════════
    mdl.update()
    print(f"    双层MIQCP: {mdl.NumVars}vars / {mdl.NumBinVars}bin / "
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

        def _v(var_dict, key=None):
            if key is not None:
                return np.array([var_dict[key][t].X for t in range(N)])
            return np.array([var_dict[t].X for t in range(N)])

        g_buy = _v(sm["g_buy"])
        p_fast = _v(sm["p_fast"])
        g_contract = _v(sm["g_contract"])
        g_spot = _v(sm["g_spot"])
        reserve_v = _v(sm["reserve"])
        ch_v = _v(sm["ch"]); dis_v = _v(sm["dis"])
        e_v = _v(sm["e"]); h_v = _v(sm["h"])
        h_elec_v = _v(sm["h_elec"])
        h_ch_grid_v = _v(sm["h_ch_grid"])
        q_heat_v = _v(sm["q_heat"])
        cs_v = _v(sm["cs"]); ss_v = _v(sm["ss"]); rs_v = _v(sm["rs"])

        p_slow_sc = np.array([[sm["p_slow"][s][t].X for t in range(N)]
                              for s in range(sm["n_scenarios"])])
        sc_pr = sm["scenario_probs"]
        p_slow_mean = np.tensordot(sc_pr, p_slow_sc, axes=(0, 0))

        user_rev = float(mp.fast_price * np.sum(p_fast) * DT
                         + mp.slow_price * np.sum(p_slow_mean) * DT)
        buy_cost = float(np.sum(buy_p * g_buy * DT))
        tc = float(np.sum((ch_v + dis_v) * THROUGHPUT_COST * DT))
        h_elec_cost = float(np.sum(buy_p * h_elec_v * DT))
        h_ch_grid_cost = float(np.sum(buy_p * h_ch_grid_v * DT))
        heat_offset = float(np.sum(q_heat_v * buy_p * DT
                                   / max(tp.heat_elec_eff, 1e-9)) * tp.heat_value_factor)

        inc_c = float(np.sum(pic_val * g_contract * DT))
        inc_s = float(np.sum(pis_val * g_spot * DT))
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
            "g_buy": g_buy, "p_fast": p_fast,
            "g_contract": g_contract, "g_spot": g_spot, "reserve": reserve_v,
            "ch": ch_v, "dis": dis_v, "e": e_v, "h": h_v,
            "h_elec": h_elec_v, "h_ch_grid": h_ch_grid_v, "q_heat": q_heat_v,
            "cs": cs_v, "ss": ss_v, "rs": rs_v,
            "p_slow_mean": p_slow_mean, "p_slow_scenarios": p_slow_sc,
            "scenario_probs": sc_pr,
            "alloc_contract": np.array([alloc_c[i][t].X for t in range(N)]),
            "alloc_spot": np.array([alloc_s[i][t].X for t in range(N)]),
            "alloc_reserve": np.array([alloc_r[i][t].X for t in range(N)]),
            "contract_del": float(np.sum(g_contract) * DT),
            "spot_del": float(np.sum(g_spot) * DT),
            "reserve_com": float(np.sum(reserve_v * mp.reserve_mask) * DT),
            "cs_kwh": cs_kwh, "ss_kwh": ss_kwh, "rs_kwh": rs_kwh,
            "buy_cost": buy_cost, "user_rev": user_rev, "tc": tc,
            "heat_offset": heat_offset,
            "heat_supply_kwh": float(np.sum(q_heat_v) * DT),
            "h_elec_cost": h_elec_cost, "h_ch_grid_cost": h_ch_grid_cost,
            "fast_ratio": float(np.sum(p_fast) / max(np.sum(station_fast[i]), 1e-9)),
            "slow_ratio": float(np.sum(sc_pr * np.sum(p_slow_sc, axis=1))
                                / max(np.sum(sc_pr * np.sum(station_slow[i], axis=1)), 1e-9)),
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