"""
DCF Valuation Model & Analysis CLI Runner
Demonstrates:
  1. Forward DCF Valuation (matches screenshot benchmark $92.12)
  2. Reverse DCF (solving market-implied growth)
  3. 3-Stage DuPont Analysis
"""

import sys
from fin_engine.dcf import (
    DCFParameters,
    calculate_dcf,
    generate_sensitivity_matrix,
)
from fin_engine.reverse_dcf import solve_implied_growth
from fin_engine.dupont import calculate_dupont_3stage, decompose_dupont_delta


def main():
    print("=" * 65)
    print("FINANCIAL VALUATION ENGINE (fin_engine)")
    print("=" * 65)

    # 1. Forward DCF (Screenshot Baseline)
    params = DCFParameters(
        base_fcff=100.0,
        growth_rate_5yr=0.08,
        wacc=0.09,
        perpetual_growth=0.025,
        shares=20.0,
        cash=50.0,
        debt=200.0
    )
    res = calculate_dcf(params)

    print("\n--- 1. FORWARD DCF VALUATION (BASELINE) ---")
    print(f"Inputs: Base FCFF=${params.base_fcff:.1f}M, 5-Yr Growth={params.growth_rate_5yr:.1%}, WACC={params.wacc:.1%}, LTG={params.perpetual_growth:.1%}")
    print(f"Capital: Shares={params.shares:.1f}M, Cash=${params.cash:.1f}M, Debt=${params.debt:.1f}M (Net Debt=${res.net_debt:.1f}M)")
    print(f"-> Intrinsic Value Per Share : ${res.intrinsic_value_per_share:.2f}")
    print(f"-> Enterprise Value          : ${res.enterprise_value:,.1f}M ({res.terminal_value_pct:.0f}% Terminal Value)")
    print(f"-> Equity Value              : ${res.equity_value:,.1f}M (EV + Cash - Debt)")

    print("\n5-Year Cash Flow Projections:")
    for p in res.projections:
        print(f"  Year {p.year}: FCFF = ${p.fcff:6.2f}M  |  Discount Factor = {1/p.discount_factor:6.4f}  |  PV = ${p.present_value:6.2f}M")
    print(f"  Sum of PV(FCFF)      = ${res.sum_pv_fcff:6.2f}M")
    print(f"  PV of Terminal Value = ${res.pv_terminal_value:6.2f}M (TV = ${res.terminal_value:,.1f}M)")

    # Sensitivity Table
    matrix_data = generate_sensitivity_matrix(params)
    print("\nSensitivity Matrix: Share Price ($) [WACC vs LTG]:")
    hdr = f"{'WACC \\ LTG':<12}" + "".join([f"{g*100:6.1f}%" for g in matrix_data["ltg_range"]])
    print(hdr)
    print("-" * len(hdr))
    for w, row in zip(matrix_data["wacc_range"], matrix_data["matrix"]):
        line = f"{w*100:5.1f}%      "
        for val in row:
            if val is not None:
                is_base = abs(w - params.wacc) < 1e-4 and abs(val - res.intrinsic_value_per_share) < 0.1
                star = "*" if is_base else " "
                line += f" ${val:5.1f}{star}"
            else:
                line += "    N/A "
        print(line)

    # 2. Reverse DCF Analysis
    print("\n--- 2. REVERSE DCF ANALYSIS (MARKET-IMPLIED GROWTH) ---")
    test_market_prices = [60.0, 75.0, 92.12, 110.0, 140.0]
    print(f"{'Target Price':<14}{'Implied 5-Yr Growth':<22}{'Implied EV':<16}{'Solver'}")
    print("-" * 65)
    for price in test_market_prices:
        rev_res = solve_implied_growth(
            target_price=price,
            base_fcff=params.base_fcff,
            wacc=params.wacc,
            perpetual_growth=params.perpetual_growth,
            shares=params.shares,
            cash=params.cash,
            debt=params.debt
        )
        print(f"${price:<13.2f}{rev_res.implied_growth_rate:<22.2%}${rev_res.target_enterprise_value:<15,.1f}M {rev_res.solver_method}")

    # 3. DuPont 3-Stage Analysis
    print("\n--- 3. 3-STAGE DUPONT DECOMPOSITION & ATTRIBUTION ---")
    dupont_prev = calculate_dupont_3stage(net_income=80.0, revenue=800.0, total_assets=1000.0, shareholders_equity=400.0)
    dupont_curr = calculate_dupont_3stage(net_income=120.0, revenue=1000.0, total_assets=1200.0, shareholders_equity=500.0)
    attr = decompose_dupont_delta(dupont_prev, dupont_curr)

    print(f"Previous Period: Margin={dupont_prev.net_profit_margin:.1%}, Turnover={dupont_prev.asset_turnover:.2f}x, Leverage={dupont_prev.equity_multiplier:.2f}x -> ROE={dupont_prev.roe:.1%}")
    print(f"Current Period : Margin={dupont_curr.net_profit_margin:.1%}, Turnover={dupont_curr.asset_turnover:.2f}x, Leverage={dupont_curr.equity_multiplier:.2f}x -> ROE={dupont_curr.roe:.1%}")
    print(f"Delta ROE      : {attr.delta_roe:+.1%}")
    print(f"  - Margin Effect   : {attr.margin_effect:+.2%} ({attr.pct_margin_contribution:.1f}% of delta)")
    print(f"  - Turnover Effect : {attr.turnover_effect:+.2%} ({attr.pct_turnover_contribution:.1f}% of delta)")
    print(f"  - Leverage Effect : {attr.leverage_effect:+.2%} ({attr.pct_leverage_contribution:.1f}% of delta)")
    print("=" * 65)


if __name__ == "__main__":
    main()
