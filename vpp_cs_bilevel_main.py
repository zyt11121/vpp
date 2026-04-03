# -*- coding: utf-8 -*-
"""
终极版双层模型 — 主入口
面向快充服务场景的、考虑电热耦合与不确定性的、VPP—储能型充电站双层协调优化模型

功能：
  1. 加载数据、生成需求
  2. 市场准入评估
  3. 冬/夏季节循环求解
  4. 结果导出（Excel + 绘图）
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vpp_cs_bilevel_base import (
    N, DT, HOURS, BASE_DIR, N_STATIONS,
    StationParam, MarketParam, ThermalParam,
    load_price_data, daily_temp_curve, kappa_winter, kappa_summer,
    build_station_data, assess_eligibility, build_dr_mask,
    setup_plot_style,
)
from vpp_cs_bilevel_upper import solve_bilevel


# ════════════════════════════════════════════════════════════════
# 无储能基准
# ════════════════════════════════════════════════════════════════
def no_bess_baseline(price_mwh, station_fast, station_slow, station_slow_probs,
                     kappa, alpha, stations, mp):
    buy_p = price_mwh / 1000
    P_g = sum(s.grid_limit_kw for s in stations)
    probs = np.asarray(station_slow_probs.mean(axis=0), dtype=float)
    probs = probs / probs.sum()
    profits = []
    for s in range(station_slow.shape[1]):
        thermal = alpha * kappa * station_fast.sum(axis=0)
        total = station_fast.sum(axis=0) + station_slow[:, s, :].sum(axis=0) + thermal
        g = np.minimum(total, P_g)
        served = np.maximum(g - thermal, 0)
        pf = np.minimum(station_fast.sum(axis=0), served)
        ps_ = np.minimum(station_slow[:, s, :].sum(axis=0), np.maximum(served - pf, 0))
        rev = (mp.fast_price * np.sum(pf) + mp.slow_price * np.sum(ps_)) * DT
        cost = np.sum(buy_p * g) * DT
        profits.append(rev - cost)
    return {"profit": float(np.dot(probs, profits))}


# ════════════════════════════════════════════════════════════════
# 季节求解
# ════════════════════════════════════════════════════════════════
def run_season(season, price_mwh, kappa, temp, alpha, beta,
               stations, station_fast, station_slow, station_slow_probs,
               eligibility, mp, tp):
    dr_mask = build_dr_mask(price_mwh, mp.dr_top_k)
    baseline = no_bess_baseline(price_mwh, station_fast, station_slow,
                                station_slow_probs, kappa, alpha, stations, mp)
    print(f"  双层协调优化 MIQCP 求解...")
    best = solve_bilevel(
        price_mwh, kappa, temp, alpha, beta,
        stations, station_fast, station_slow, station_slow_probs,
        eligibility, mp, tp, dr_mask)
    if best is not None:
        best["baseline"] = baseline
        best["system_profit_vs_baseline"] = best["system_profit"] - baseline["profit"]
    return best


# ════════════════════════════════════════════════════════════════
# 结果导出
# ════════════════════════════════════════════════════════════════
def export_results(season, best, price_mwh, temp, kappa, station_fast,
                   station_slow_probs, alpha, beta, rep_day, mp):
    srs = best["station_results"]

    def agg(k):
        return sum(r[k] for r in srs)

    # ── 站级汇总 ──
    rows = []
    for r in srs:
        rows.append({
            "station_id": r["station_id"],
            "user_rev": r["user_rev"],
            "incentive_total": r["incentive_total"],
            "buy_cost": r["buy_cost"],
            "tc": r["tc"],
            "thermal_cost": r["thermal_cost"],
            "penalty_cost": r["penalty_cost"],
            "station_profit": r["station_profit"],
            "contract_del_kwh": r["contract_del"],
            "spot_del_kwh": r["spot_del"],
            "reserve_com_kwh": r["reserve_com"],
            "fast_ratio": r["fast_ratio"],
            "slow_ratio": r["slow_ratio"],
            "heat_supply_kwh": r["heat_supply_kwh"],
            "h_elec_cost": r["h_elec_cost"],
            "h_ch_grid_cost": r["h_ch_grid_cost"],
        })
    pd.DataFrame(rows).to_excel(
        os.path.join(BASE_DIR, f"{season}_bilevel_站点结果.xlsx"), index=False)

    # ── 时序明细 ──
    pd.DataFrame({
        "time_h": HOURS,
        "price_MWh": price_mwh,
        "temp_C": temp,
        "kappa": kappa,
        "pi_contract": best["pi_contract"],
        "pi_spot": best["pi_spot"],
        "pi_reserve": best["pi_reserve"],
        "contract_bid_kW": best["contract_curve"],
        "spot_bid_kW": best["spot_offer"],
        "reserve_bid_kW": best["reserve_target"],
        "actual_contract": agg("g_contract"),
        "actual_spot": agg("g_spot"),
        "actual_reserve": agg("reserve"),
        "grid_buy_kW": agg("g_buy"),
        "fast_kW": agg("p_fast"),
        "slow_kW": agg("p_slow_mean"),
        "bess_ch_kW": agg("ch"),
        "bess_dis_kW": agg("dis"),
        "bess_e_kWh": agg("e"),
        "heat_kWh": agg("h"),
        "h_elec_kW": agg("h_elec"),
        "h_ch_grid_kW": agg("h_ch_grid"),
        "q_heat_kW": agg("q_heat"),
        "cs_kW": agg("cs"),
        "ss_kW": agg("ss"),
        "rs_kW": agg("rs"),
    }).to_excel(os.path.join(BASE_DIR,
        f"{season}_a{alpha:.3f}_b{beta:.3f}_bilevel明细.xlsx"), index=False)

    # ── 收益汇总 ──
    pd.DataFrame([{
        "season": season, "rep_day": str(rep_day),
        "alpha": alpha, "beta": beta,
        "contract_ratio": best["contract_ratio"],
        "spot_ratio": best["spot_ratio"],
        "reserve_kw": best["reserve_kw"],
        "vpp_profit": best["vpp_profit"],
        "stations_total_profit": best["stations_total_profit"],
        "system_profit": best["system_profit"],
        "incentive_cost": best["incentive_cost"],
        "user_rev": best["user_rev"],
        "contract_rev": best["contract_rev"],
        "spot_rev": best["spot_rev"],
        "reserve_rev": best["reserve_rev"],
        "dr_rev": best["dr_rev"],
        "buy_cost": best["buy_cost"],
        "tc": best["throughput_cost"],
        "penalty": best["penalty"],
        "pi_contract_mean": float(np.mean(best["pi_contract"])),
        "pi_spot_mean": float(np.mean(best["pi_spot"])),
        "pi_reserve_mean": float(np.mean(best["pi_reserve"])),
        "baseline_profit": best["baseline"]["profit"],
        "system_vs_baseline": best["system_profit_vs_baseline"],
        "status": best["milp_status"],
        "n_vars": best["n_total_vars"],
        "n_bin": best["n_binary_vars"],
        "n_qconstr": best["n_qconstr"],
    }]).to_excel(os.path.join(BASE_DIR,
        f"{season}_bilevel_收益汇总.xlsx"), index=False)

    # ── 绘图 ──
    setup_plot_style()
    agg_g_buy = agg("g_buy")
    agg_ch = agg("ch"); agg_dis = agg("dis")
    agg_e = agg("e"); agg_h = agg("h")
    agg_contract = agg("g_contract")
    agg_spot = agg("g_spot")
    agg_reserve = agg("reserve")
    agg_fast = agg("p_fast")
    agg_slow = agg("p_slow_mean")
    net_bess = agg_dis - agg_ch

    fig, axes = plt.subplots(6, 1, figsize=(15.6, 19.2), sharex=True,
        gridspec_kw={"height_ratios": [1, 1, 1.15, 1, 1, 1.1], "hspace": 0.28})

    title = (f"{season.capitalize()} | {rep_day} | "
             f"VPP={best['vpp_profit']:.1f} | 站端={best['stations_total_profit']:.1f} | "
             f"System={best['system_profit']:.1f}")
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.99)

    def _style(ax, t=None, yl=None):
        if t: ax.set_title(t, loc="left", fontweight="bold", pad=5)
        if yl: ax.set_ylabel(yl)
        ax.grid(True, axis="y", alpha=0.18, linestyle="--")
        ax.set_xlim(HOURS[0], HOURS[-1])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 1) 电价+温度
    ax = axes[0]
    ax.fill_between(HOURS, 0, price_mwh, color="#D9D9D9", alpha=0.5)
    ax.plot(HOURS, price_mwh, color="#2F2F2F", lw=1.8, label="电价")
    ax2 = ax.twinx()
    ax2.plot(HOURS, temp, color="#D62728", lw=1.4, label="温度")
    _style(ax, "1. 外部环境", "元/MWh")

    # 2) 激励价格 vs 市场价格
    ax = axes[1]
    cup_plot = np.full(N, mp.contract_price / 1000.0)
    sup_plot = np.maximum(price_mwh - mp.spot_discount, 0) / 1000
    ax.plot(HOURS, cup_plot, color="#4C78A8", lw=1.2, ls=":", label="合约市场价")
    ax.plot(HOURS, best["pi_contract"], color="#4C78A8", lw=2.0, label="π_contract")
    ax.plot(HOURS, sup_plot, color="#2CA02C", lw=1.2, ls=":", label="现货市场价")
    ax.plot(HOURS, best["pi_spot"], color="#2CA02C", lw=2.0, label="π_spot")
    ax.plot(HOURS, best["pi_reserve"], color="#D62728", lw=2.0, label="π_reserve")
    _style(ax, "2. 激励价格 vs 市场价格 (元/kWh)", "元/kWh")
    ax.legend(ncol=5, loc="upper left", frameon=False, fontsize=8)

    # 3) 功率调度
    ax = axes[2]
    ax.plot(HOURS, agg_fast, color="#FF7F0E", lw=2, label="快充")
    ax.plot(HOURS, agg_slow, color="#2CA02C", lw=1.8, label="慢充")
    ax.plot(HOURS, agg_g_buy, color="#1F77B4", lw=2, label="购电")
    pos = np.where(net_bess > 1e-9, net_bess, 0)
    neg = np.where(net_bess < -1e-9, net_bess, 0)
    ax.bar(HOURS, pos, width=0.17, color="#E9A46A", alpha=0.8, label="净放电")
    ax.bar(HOURS, neg, width=0.17, color="#8ECAE6", alpha=0.8, label="净充电")
    _style(ax, "3. 功率调度", "kW")
    ax.legend(ncol=5, loc="upper left", frameon=False, fontsize=8)

    # 4) 储能状态
    ax = axes[3]
    ax.plot(HOURS, agg_e, color="#2B6CB0", lw=2, label="电储能")
    ax2 = ax.twinx()
    ax2.plot(HOURS, agg_h, color="#C2185B", lw=1.8, label="热储能")
    _style(ax, "4. 储能状态", "kWh")

    # 5) 热管理
    ax = axes[4]
    ax.plot(HOURS, agg("q_heat"), color="#7B2CBF", lw=2, label="热储能供热")
    ax.plot(HOURS, agg("h_elec"), color="#D62728", lw=1.4, ls="--", label="电加热")
    ax.plot(HOURS, agg("h_ch_grid"), color="#BC6C25", lw=1.4, ls=":", label="主动储热")
    _style(ax, "5. 热管理", "kW")
    ax.legend(ncol=3, loc="upper left", frameon=False, fontsize=8)

    # 6) 市场申报与兑现
    ax = axes[5]
    ax.plot(HOURS, best["market_power_profile"], color="#A9A9A9", lw=1.2, ls=":", label="上限")
    ax.plot(HOURS, best["contract_curve"], color="#4C78A8", lw=1.2, ls="--", label="合约申报")
    ax.plot(HOURS, agg_contract, color="#4C78A8", lw=2, label="合约兑现")
    ax.plot(HOURS, best["spot_offer"], color="#2CA02C", lw=1.2, ls="--", label="现货申报")
    ax.plot(HOURS, agg_spot, color="#2CA02C", lw=2, label="现货兑现")
    ax.plot(HOURS, best["reserve_target"], color="#D62728", lw=1.2, ls="--", label="备用申报")
    ax.plot(HOURS, agg_reserve, color="#D62728", lw=2, label="备用兑现")
    _style(ax, "6. 市场申报与兑现", "kW")
    ax.legend(ncol=4, loc="upper left", frameon=False, fontsize=8)
    ax.set_xlabel("时间")

    xticks = np.arange(0, 24, 2)
    axes[-1].set_xticks(xticks)
    axes[-1].set_xticklabels([f"{int(x):02d}:00" for x in xticks])

    plt.tight_layout(rect=[0.02, 0.02, 0.98, 0.97])
    plt.savefig(os.path.join(BASE_DIR,
        f"{season}_a{alpha:.3f}_b{beta:.3f}_bilevel.png"),
        dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"  [{season}] 图表已保存")


# ════════════════════════════════════════════════════════════════
# 主函数
# ════════════════════════════════════════════════════════════════
def main():
    mp = MarketParam()
    tp = ThermalParam()
    stations = [StationParam(station_id=i + 1) for i in range(N_STATIONS)]

    print("=== 读取电价数据 ===")
    wd, sd, pw, ps = load_price_data()
    print(f"  冬季典型日: {wd}  夏季典型日: {sd}")

    tw = daily_temp_curve(10.2, 18.5)
    ts = daily_temp_curve(25.6, 31.7)
    kw = kappa_winter(tw)
    ks = kappa_summer(ts)

    print("\n=== 生成充电需求数据 ===")
    station_fast, station_slow, station_slow_probs = build_station_data()
    print(f"  快充 {station_fast.shape}  慢充 {station_slow.shape}")

    print("\n=== 市场准入评估 ===")
    elig = assess_eligibility(stations, mp)
    for k, v in elig.items():
        print(f"  {k}: {v}")
    pd.DataFrame([elig]).to_excel(
        os.path.join(BASE_DIR, "bilevel_市场准入评估.xlsx"), index=False)

    all_sum = []
    for alpha, beta in [(0.05, 0.018)]:
        for season, price, kap, temp, rep_day in [
            ("winter", pw, kw, tw, wd),
            ("summer", ps, ks, ts, sd),
        ]:
            print(f"\n{'=' * 60}")
            print(f"季节={season}  α={alpha}  β={beta}")
            print(f"{'=' * 60}")

            best = run_season(
                season, price, kap, temp, alpha, beta,
                stations, station_fast, station_slow,
                station_slow_probs, elig, mp, tp)

            if best is None:
                print(f"  {season}: 无可行解")
                continue

            print(f"\n  VPP利润:     {best['vpp_profit']:.1f} 元")
            print(f"  站端总利润:  {best['stations_total_profit']:.1f} 元")
            print(f"  系统总利润:  {best['system_profit']:.1f} 元")
            print(f"  激励成本:    {best['incentive_cost']:.1f} 元")
            print(f"  vs 基准:     +{best['system_profit_vs_baseline']:.1f} 元")
            print(f"  π_contract均值: {np.mean(best['pi_contract']):.4f}")
            print(f"  π_spot均值:     {np.mean(best['pi_spot']):.4f}")
            print(f"  π_reserve均值:  {np.mean(best['pi_reserve']):.4f}")

            # 打印各站利润
            for r in best["station_results"]:
                print(f"  站{r['station_id']}: 利润={r['station_profit']:.1f} "
                      f"(充电={r['user_rev']:.1f} 激励={r['incentive_total']:.1f} "
                      f"购电=-{r['buy_cost']:.1f} 热管理=-{r['thermal_cost']:.1f})")

            export_results(season, best, price, temp, kap,
                           station_fast, station_slow_probs,
                           alpha, beta, rep_day, mp)

            all_sum.append({
                "season": season, "alpha": alpha, "beta": beta,
                "vpp_profit": best["vpp_profit"],
                "stations_total_profit": best["stations_total_profit"],
                "system_profit": best["system_profit"],
                "incentive_cost": best["incentive_cost"],
                "system_vs_baseline": best["system_profit_vs_baseline"],
                "status": best["milp_status"],
            })

    if all_sum:
        pd.DataFrame(all_sum).to_excel(
            os.path.join(BASE_DIR, "bilevel_总汇总.xlsx"), index=False)
    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()