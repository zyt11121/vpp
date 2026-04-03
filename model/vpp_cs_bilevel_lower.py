# -*- coding: utf-8 -*-
"""
真正双层模型 — 下层充电站KKT嵌入

在上层Gurobi模型中，为每个充电站嵌入完整KKT条件:
  原始可行性 + 对偶可行性 + 驻点条件 + 互补松弛(Big-M线性化)
"""
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from model.vpp_cs_bilevel_base import (
    N, DT, BIG_M, THROUGHPUT_COST, KKT_DUAL_UB,
    eta_from_temp, soc_range_from_temp, power_factor_from_temp,
)


def _comp_slack(mdl, lam, slack_expr, rhs, prefix, t):
    """互补松弛线性化: lam*(rhs - expr) = 0"""
    z = mdl.addVar(vtype=GRB.BINARY, name=f"{prefix}_z_{t}")
    mdl.addConstr(lam <= BIG_M * z, name=f"{prefix}_c1_{t}")
    mdl.addConstr(rhs - slack_expr <= BIG_M * (1 - z), name=f"{prefix}_c2_{t}")


def _bound_comp(mdl, nu, var_expr, bound_val, prefix, t, is_upper):
    """变量界互补松弛线性化"""
    z = mdl.addVar(vtype=GRB.BINARY, name=f"{prefix}_z_{t}")
    mdl.addConstr(nu <= BIG_M * z, name=f"{prefix}_c1_{t}")
    if is_upper:
        mdl.addConstr(bound_val - var_expr <= BIG_M * (1 - z), name=f"{prefix}_c2_{t}")
    else:
        mdl.addConstr(var_expr - bound_val <= BIG_M * (1 - z), name=f"{prefix}_c2_{t}")


def build_lower_kkt(mdl, station, price_mwh, fast_kw, slow_expected,
                    kappa, temp, alpha, beta, mp, tp,
                    pi_c, pi_s, pi_r,
                    alloc_c, alloc_s, alloc_r,
                    station_idx):
    """在Gurobi模型中嵌入第station_idx个充电站的完整KKT条件。"""
    sid = station_idx
    bp = price_mwh / 1000.0
    eta_c_t = eta_from_temp(temp, station.eta_c)
    eta_d_t = eta_from_temp(temp, station.eta_d)
    smin_t, smax_t = soc_range_from_temp(temp, station.soc_min, station.soc_max)
    pf_t = power_factor_from_temp(temp, tp.temp_power_ref_c, tp.temp_power_slope_per_c, tp.temp_power_floor)
    Ecap = station.bess_energy_kwh
    E0 = station.soc0 * Ecap
    Pgrid = station.grid_limit_kw
    Pbess = station.bess_power_kw
    fp = mp.fast_price
    slp = mp.slow_price
    TC = THROUGHPUT_COST
    cp_pen = mp.contract_penalty
    sp_pen = mp.spot_penalty
    ap_pen = mp.as_penalty
    ehc = tp.heat_charge_eff
    ehe = tp.heat_elec_eff
    hloss = tp.heat_loss

    # 变量界数组
    ub = {
        "gb": np.full(N, Pgrid), "pf": np.maximum(fast_kw, 0.0),
        "ps": np.maximum(slow_expected, 0.0), "gc": np.full(N, Pgrid),
        "gs": np.full(N, Pgrid), "rv": Pbess * pf_t,
        "ch": Pbess * pf_t, "di": Pbess * pf_t,
        "e": smax_t * Ecap, "h": np.full(N, tp.heat_max_kwh),
        "he": np.full(N, Pgrid), "hc": np.full(N, tp.heat_charge_max_kw),
        "qh": np.full(N, tp.heat_max_kwh / DT),
        "cs": np.full(N, Pgrid), "ss": np.full(N, Pgrid), "rs": np.full(N, Pgrid),
    }
    lb = {k: np.zeros(N) for k in ub}
    lb["e"] = smin_t * Ecap

    # ═══ 原始变量 ═══
    x = {}
    for k in ub:
        x[k] = mdl.addVars(N, lb=-GRB.INFINITY, ub=GRB.INFINITY, name=f"s{sid}_{k}")
    # 简写
    gb, pf_, ps, gc, gs, rv = x["gb"], x["pf"], x["ps"], x["gc"], x["gs"], x["rv"]
    ch, di, ev, hv = x["ch"], x["di"], x["e"], x["h"]
    he, hc, qh = x["he"], x["hc"], x["qh"]
    csv, ssv, rsv = x["cs"], x["ss"], x["rs"]

    # ═══ 对偶变量 ═══
    mu_bal = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_bal")
    mu_soc = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_soc")
    mu_se = mdl.addVar(lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_se")
    mu_ht = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_ht")
    mu_hb = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_hb")
    mu_cd = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_cd")
    mu_sd = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_sd")
    mu_rd = mdl.addVars(N, lb=-KKT_DUAL_UB, ub=KKT_DUAL_UB, name=f"s{sid}_mu_rd")
    la_gr = mdl.addVars(N, lb=0, ub=KKT_DUAL_UB, name=f"s{sid}_la_gr")
    la_mc = mdl.addVars(N, lb=0, ub=KKT_DUAL_UB, name=f"s{sid}_la_mc")
    la_rs = mdl.addVars(N, lb=0, ub=KKT_DUAL_UB, name=f"s{sid}_la_rs")
    la_dr = mdl.addVars(N, lb=0, ub=KKT_DUAL_UB, name=f"s{sid}_la_dr")
    nulb = {k: mdl.addVars(N, lb=0, ub=KKT_DUAL_UB, name=f"s{sid}_nl_{k}") for k in ub}
    nuub = {k: mdl.addVars(N, lb=0, ub=KKT_DUAL_UB, name=f"s{sid}_nu_{k}") for k in ub}

    # ═══ (1) 原始可行性 ═══
    for t in range(N):
        ec = float(eta_c_t[t]); ed = float(eta_d_t[t])
        kap = float(kappa[t]); pfv = float(Pbess * pf_t[t])
        sminE = float(smin_t[t] * Ecap)

        # A1 bal
        mdl.addConstr(gb[t] + di[t] - pf_[t] - ps[t] - he[t] - hc[t] - ch[t] - gc[t] - gs[t] == 0, name=f"s{sid}_A1_{t}")
        # A2 soc
        if t == 0:
            mdl.addConstr(ev[t] - ec*DT*ch[t] + DT/ed*di[t] == E0, name=f"s{sid}_A2_{t}")
        else:
            mdl.addConstr(ev[t] - ev[t-1] - ec*DT*ch[t] + DT/ed*di[t] == 0, name=f"s{sid}_A2_{t}")
        # A4 heat
        if t == 0:
            mdl.addConstr(hv[t] - beta*DT*pf_[t] - ehc*DT*hc[t] + DT*qh[t] == tp.heat_init_kwh, name=f"s{sid}_A4_{t}")
        else:
            mdl.addConstr(hv[t] - (1-hloss)*hv[t-1] - beta*DT*pf_[t] - ehc*DT*hc[t] + DT*qh[t] == 0, name=f"s{sid}_A4_{t}")
        # A5 hbal
        mdl.addConstr(qh[t] + ehe*he[t] - alpha*kap*pf_[t] == 0, name=f"s{sid}_A5_{t}")
        # A6-A8
        mdl.addConstr(gc[t] + csv[t] == alloc_c[t], name=f"s{sid}_A6_{t}")
        mdl.addConstr(gs[t] + ssv[t] == alloc_s[t], name=f"s{sid}_A7_{t}")
        mdl.addConstr(rv[t] + rsv[t] == alloc_r[t], name=f"s{sid}_A8_{t}")
        # B1-B4
        mdl.addConstr(gb[t] + gc[t] + gs[t] <= Pgrid, name=f"s{sid}_B1_{t}")
        mdl.addConstr(gc[t] + gs[t] + rv[t] - di[t] - gb[t] <= 0, name=f"s{sid}_B2_{t}")
        mdl.addConstr(rv[t]*DT/ed - ev[t] + sminE <= 0, name=f"s{sid}_B3_{t}")
        mdl.addConstr(di[t] + rv[t] <= pfv, name=f"s{sid}_B4_{t}")
        # 变量界约束
        for k in ub:
            mdl.addConstr(x[k][t] <= float(ub[k][t]), name=f"s{sid}_ub_{k}_{t}")
            mdl.addConstr(x[k][t] >= float(lb[k][t]), name=f"s{sid}_lb_{k}_{t}")

    # A3 终端SOC
    mdl.addConstr(ev[N-1] == E0, name=f"s{sid}_A3")

    # ═══ (3) 驻点条件 ═══
    for t in range(N):
        ec = float(eta_c_t[t]); ed = float(eta_d_t[t])
        kap = float(kappa[t])

        # gb: c=bp*DT, A1:+1, B1:+1, B2:-1
        mdl.addConstr(bp[t]*DT + mu_bal[t] + la_gr[t] - la_mc[t] + nuub["gb"][t] - nulb["gb"][t] == 0, name=f"s{sid}_S_gb_{t}")
        # pf: c=-fp*DT, A1:-1, A4:-β*DT, A5:-α*κ
        mdl.addConstr(-fp*DT - mu_bal[t] - beta*DT*mu_ht[t] - alpha*kap*mu_hb[t] + nuub["pf"][t] - nulb["pf"][t] == 0, name=f"s{sid}_S_pf_{t}")
        # ps: c=-slp*DT, A1:-1
        mdl.addConstr(-slp*DT - mu_bal[t] + nuub["ps"][t] - nulb["ps"][t] == 0, name=f"s{sid}_S_ps_{t}")
        # gc: c=-π_c*DT, A1:-1, A6:+1, B1:+1, B2:+1
        mdl.addConstr(-pi_c[t]*DT - mu_bal[t] + mu_cd[t] + la_gr[t] + la_mc[t] + nuub["gc"][t] - nulb["gc"][t] == 0, name=f"s{sid}_S_gc_{t}")
        # gs: c=-π_s*DT, A1:-1, A7:+1, B1:+1, B2:+1
        mdl.addConstr(-pi_s[t]*DT - mu_bal[t] + mu_sd[t] + la_gr[t] + la_mc[t] + nuub["gs"][t] - nulb["gs"][t] == 0, name=f"s{sid}_S_gs_{t}")
        # rv: c=-π_r*DT, A8:+1, B2:+1, B3:+DT/ed, B4:+1
        mdl.addConstr(-pi_r[t]*DT + mu_rd[t] + la_mc[t] + DT/ed*la_rs[t] + la_dr[t] + nuub["rv"][t] - nulb["rv"][t] == 0, name=f"s{sid}_S_rv_{t}")
        # ch: c=TC*DT, A1:-1, A2:-ec*DT
        mdl.addConstr(TC*DT - mu_bal[t] - ec*DT*mu_soc[t] + nuub["ch"][t] - nulb["ch"][t] == 0, name=f"s{sid}_S_ch_{t}")
        # di: c=TC*DT, A1:+1, A2:+DT/ed, B2:-1, B4:+1
        mdl.addConstr(TC*DT + mu_bal[t] + DT/ed*mu_soc[t] - la_mc[t] + la_dr[t] + nuub["di"][t] - nulb["di"][t] == 0, name=f"s{sid}_S_di_{t}")
        # e: c=0, A2_t:+1, A2_{t+1}:-1(if t<N-1), A3(if t==N-1):+1, B3:-1
        grad_e = gp.LinExpr()
        grad_e += mu_soc[t]
        if t < N - 1:
            grad_e -= mu_soc[t+1]
        if t == N - 1:
            grad_e += mu_se
        grad_e -= la_rs[t]
        grad_e += nuub["e"][t] - nulb["e"][t]
        mdl.addConstr(grad_e == 0, name=f"s{sid}_S_e_{t}")
        # h: c=0, A4_t:+1, A4_{t+1}:-(1-hloss)(if t<N-1)
        grad_h = gp.LinExpr()
        grad_h += mu_ht[t]
        if t < N - 1:
            grad_h -= (1 - hloss) * mu_ht[t+1]
        grad_h += nuub["h"][t] - nulb["h"][t]
        mdl.addConstr(grad_h == 0, name=f"s{sid}_S_h_{t}")
        # he: c=bp*DT, A1:-1, A5:+ehe
        mdl.addConstr(bp[t]*DT - mu_bal[t] + ehe*mu_hb[t] + nuub["he"][t] - nulb["he"][t] == 0, name=f"s{sid}_S_he_{t}")
        # hc: c=bp*DT, A1:-1, A4:-ehc*DT
        mdl.addConstr(bp[t]*DT - mu_bal[t] - ehc*DT*mu_ht[t] + nuub["hc"][t] - nulb["hc"][t] == 0, name=f"s{sid}_S_hc_{t}")
        # qh: c=0, A4:+DT, A5:+1
        mdl.addConstr(DT*mu_ht[t] + mu_hb[t] + nuub["qh"][t] - nulb["qh"][t] == 0, name=f"s{sid}_S_qh_{t}")
        # cs: c=cp_pen*DT, A6:+1
        mdl.addConstr(cp_pen*DT + mu_cd[t] + nuub["cs"][t] - nulb["cs"][t] == 0, name=f"s{sid}_S_cs_{t}")
        # ss: c=sp_pen*DT, A7:+1
        mdl.addConstr(sp_pen*DT + mu_sd[t] + nuub["ss"][t] - nulb["ss"][t] == 0, name=f"s{sid}_S_ss_{t}")
        # rs: c=ap_pen*DT, A8:+1
        mdl.addConstr(ap_pen*DT + mu_rd[t] + nuub["rs"][t] - nulb["rs"][t] == 0, name=f"s{sid}_S_rs_{t}")

    # ═══ (4) 互补松弛 (Big-M线性化) ═══
    for t in range(N):
        ed = float(eta_d_t[t])
        pfv = float(Pbess * pf_t[t])
        sminE = float(smin_t[t] * Ecap)
        # B1: la_gr * (Pgrid - gb - gc - gs) = 0
        _comp_slack(mdl, la_gr[t], gb[t] + gc[t] + gs[t], Pgrid, f"s{sid}_cB1", t)
        # B2: la_mc * (0 - gc - gs - rv + di + gb) = 0
        _comp_slack(mdl, la_mc[t], gc[t] + gs[t] + rv[t] - di[t] - gb[t], 0.0, f"s{sid}_cB2", t)
        # B3: la_rs * (-rv*DT/ed + e - sminE) = 0
        _comp_slack(mdl, la_rs[t], rv[t]*DT/ed - ev[t] + sminE, 0.0, f"s{sid}_cB3", t)
        # B4: la_dr * (pfv - di - rv) = 0
        _comp_slack(mdl, la_dr[t], di[t] + rv[t], pfv, f"s{sid}_cB4", t)
        # 变量界互补
        for k in ub:
            _bound_comp(mdl, nulb[k][t], x[k][t], float(lb[k][t]), f"s{sid}_clb_{k}", t, False)
            _bound_comp(mdl, nuub[k][t], x[k][t], float(ub[k][t]), f"s{sid}_cub_{k}", t, True)

    return {
        "g_buy": gb, "p_fast": pf_, "p_slow": ps,
        "g_contract": gc, "g_spot": gs, "reserve": rv,
        "ch": ch, "dis": di, "e": ev, "h": hv,
        "h_elec": he, "h_ch_grid": hc, "q_heat": qh,
        "cs": csv, "ss": ssv, "rs": rsv,
        "eta_c_t": eta_c_t, "eta_d_t": eta_d_t,
        "soc_min_t": smin_t, "soc_max_t": smax_t,
        "p_factor_t": pf_t,
    }