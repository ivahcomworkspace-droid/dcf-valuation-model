"""
WACC & CAPM Cost of Capital Module (fin_engine.wacc)
Pure mathematical domain model for calculating Cost of Equity (CAPM)
and Weighted Average Cost of Capital (WACC) without external side effects or I/O.
"""

from typing import Optional, Dict


def calculate_cost_of_equity(
    risk_free_rate: float,
    beta: float,
    equity_risk_premium: float = 0.055
) -> float:
    """
    Computes Cost of Equity (Ke) via the Capital Asset Pricing Model (CAPM):
        Ke = Rf + Beta * ERP

    :param risk_free_rate: Risk-free rate as decimal (e.g. 0.042 for 4.2%)
    :param beta: Equity beta measuring systematic market covariance
    :param equity_risk_premium: Expected excess market return over risk-free asset (default 5.5%)
    :return: Cost of Equity as decimal
    """
    if beta < 0:
        raise ValueError("Beta must be non-negative.")
    if equity_risk_premium <= 0:
        raise ValueError("Equity Risk Premium must be strictly positive.")

    return risk_free_rate + (beta * equity_risk_premium)


def calculate_wacc(
    equity_value: float,
    debt_value: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float = 0.21
) -> float:
    """
    Computes Weighted Average Cost of Capital (WACC):
        WACC = (E / (E + D)) * Ke + (D / (E + D)) * Kd * (1 - t)

    :param equity_value: Market capitalization or equity value ($M)
    :param debt_value: Total debt balance ($M)
    :param cost_of_equity: Expected return on equity (Ke) as decimal
    :param cost_of_debt: Pre-tax cost of debt (Kd) as decimal
    :param tax_rate: Marginal or effective corporate tax rate as decimal (default 21%)
    :return: WACC as decimal (e.g. 0.09 for 9.0%)
    """
    if equity_value < 0 or debt_value < 0:
        raise ValueError("Equity and Debt values must be non-negative.")

    total_cap = equity_value + debt_value
    if total_cap <= 0:
        return max(0.05, min(0.20, cost_of_equity))

    w_e = equity_value / total_cap
    w_d = debt_value / total_cap

    effective_tax = max(0.0, min(0.50, tax_rate))
    after_tax_kd = cost_of_debt * (1.0 - effective_tax)

    wacc = (w_e * cost_of_equity) + (w_d * after_tax_kd)
    return max(0.04, min(0.25, wacc))


def calculate_dynamic_wacc(
    price: float,
    shares: float,
    debt: float,
    beta: float = 1.0,
    currency: str = "USD",
    tax_rate: Optional[float] = None,
    interest_expense: Optional[float] = None,
    risk_free_rate: Optional[float] = None,
    equity_risk_premium: float = 0.055
) -> Dict[str, float]:
    """
    High-level orchestrator that resolves currency-specific benchmarks,
    derives capital structure weights, and solves dynamic WACC.

    :return: Dictionary containing WACC, Ke, Kd, weights, and inputs.
    """
    # 1. Currency benchmark resolution for Risk-Free Rate
    curr_upper = (currency or "USD").strip().upper()
    if risk_free_rate is None:
        if curr_upper == "INR":
            rf = 0.068  # 10Y Indian Government Benchmark Bond yield (~6.8%)
        elif curr_upper in ("EUR", "GBP", "CHF"):
            rf = 0.032
        elif curr_upper == "JPY":
            rf = 0.015
        else:
            rf = 0.042  # 10Y US Treasury yield (~4.2%)
    else:
        rf = risk_free_rate

    # 2. Cost of Equity (CAPM)
    effective_beta = max(0.2, min(3.5, beta if beta is not None else 1.0))
    ke = calculate_cost_of_equity(rf, effective_beta, equity_risk_premium)

    # 3. Capital weights
    equity_val = max(0.0, price * shares)
    debt_val = max(0.0, debt)
    total_val = equity_val + debt_val
    w_e = equity_val / total_val if total_val > 0 else 1.0
    w_d = debt_val / total_val if total_val > 0 else 0.0

    # 4. Cost of Debt (Kd)
    if interest_expense is not None and debt_val > 0 and interest_expense > 0:
        raw_kd = interest_expense / debt_val
        kd = max(0.03, min(0.14, raw_kd))
    else:
        kd = rf + 0.015  # Default ~150 bps corporate spread over risk-free

    # 5. Effective Tax Rate
    t = tax_rate if (tax_rate is not None and 0.0 <= tax_rate <= 0.50) else (0.25 if curr_upper == "INR" else 0.21)

    wacc = calculate_wacc(equity_val, debt_val, ke, kd, t)

    return {
        "wacc": wacc,
        "cost_of_equity": ke,
        "cost_of_debt": kd,
        "equity_weight": w_e,
        "debt_weight": w_d,
        "risk_free_rate": rf,
        "beta": effective_beta,
        "tax_rate": t,
    }
