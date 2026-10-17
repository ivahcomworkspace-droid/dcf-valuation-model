"""
Reverse DCF Engine
Calculates the market-implied 5-Year FCFF / Revenue growth rate given current share price.
Uses Brent's root-finding method (scipy.optimize.brentq) with a robust binary-search fallback.
Completely isolated from network calls.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import math

try:
    from scipy.optimize import brentq
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from fin_engine.dcf import DCFParameters, calculate_dcf


@dataclass(frozen=True)
class ReverseDCFResult:
    """Output metrics of a completed Reverse DCF run."""
    target_price: float
    implied_growth_rate: float        # Solved annual growth rate (decimal, e.g. 0.08 for 8.0%)
    target_equity_value: float        # $M
    target_enterprise_value: float    # $M
    base_fcff: float
    wacc: float
    perpetual_growth: float
    iterations: int
    solver_method: str
    validation_price: float           # Re-evaluated price using implied growth to verify accuracy
    pricing_error: float              # |validation_price - target_price|


def _objective_function(
    g: float,
    target_price: float,
    base_fcff: float,
    wacc: float,
    perpetual_growth: float,
    shares: float,
    cash: float,
    debt: float
) -> float:
    """Evaluates (IntrinsicValue(g) - target_price). Root occurs where this equals 0."""
    params = DCFParameters(
        base_fcff=base_fcff,
        growth_rate_5yr=g,
        wacc=wacc,
        perpetual_growth=perpetual_growth,
        shares=shares,
        cash=cash,
        debt=debt
    )
    result = calculate_dcf(params)
    return result.intrinsic_value_per_share - target_price


def _solve_binary_search(
    target_price: float,
    base_fcff: float,
    wacc: float,
    perpetual_growth: float,
    shares: float,
    cash: float,
    debt: float,
    low: float,
    high: float,
    tol: float = 1e-6,
    max_iter: int = 100
) -> Tuple[float, int]:
    """Fallback numerical solver using binary search (bisection)."""
    f_low = _objective_function(low, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)
    f_high = _objective_function(high, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)

    if f_low * f_high > 0:
        raise ValueError(
            f"Root is not bracketed: f({low:.2%})={f_low:.4f}, f({high:.2%})={f_high:.4f}. "
            "Target price may be outside solvable range."
        )

    for i in range(1, max_iter + 1):
        mid = (low + high) / 2.0
        f_mid = _objective_function(mid, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)
        if abs(f_mid) < tol or (high - low) / 2.0 < tol:
            return mid, i
        if f_low * f_mid <= 0:
            high = mid
            f_high = f_mid
        else:
            low = mid
            f_low = f_mid

    return mid, max_iter


def solve_implied_growth(
    target_price: float,
    base_fcff: float,
    wacc: float,
    perpetual_growth: float,
    shares: float,
    cash: float = 0.0,
    debt: float = 0.0,
    bracket: Tuple[float, float] = (-0.95, 3.00),
    tol: float = 1e-6,
    force_fallback: bool = False
) -> ReverseDCFResult:
    """
    Solves for the 5-Year FCFF growth rate implied by a given target market share price.

    Args:
        target_price: Current market share price ($)
        base_fcff: Base Free Cash Flow to Firm ($M)
        wacc: Weighted Average Cost of Capital (decimal, e.g. 0.09)
        perpetual_growth: Terminal growth rate (decimal, e.g. 0.025)
        shares: Shares outstanding (Millions)
        cash: Cash balance ($M)
        debt: Total debt ($M)
        bracket: Search interval for growth rate (default -95% to +300%)
        tol: Convergence tolerance ($)
        force_fallback: If True, uses binary search even if scipy is installed.

    Returns:
        ReverseDCFResult with implied growth rate and verification metrics.
    """
    if target_price <= 0:
        raise ValueError(f"Target share price must be strictly positive, got ${target_price:.2f}")
    if shares <= 0:
        raise ValueError(f"Shares outstanding must be positive, got {shares}")
    if wacc <= perpetual_growth:
        raise ValueError(f"WACC ({wacc:.2%}) must exceed Perpetual Growth ({perpetual_growth:.2%})")

    target_equity_value = target_price * shares
    target_enterprise_value = target_equity_value + debt - cash

    if target_enterprise_value <= 0 and base_fcff > 0:
        raise ValueError(
            f"Target Enterprise Value (${target_enterprise_value:.2f}M) is non-positive. "
            "A company with positive base cash flows cannot produce a non-positive EV."
        )

    low, high = bracket
    f_low = _objective_function(low, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)
    f_high = _objective_function(high, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)

    # Dynamic bracket expansion if initial bracket doesn't bound the root
    if f_low > 0 and low > -0.99:
        low = -0.99
        f_low = _objective_function(low, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)
    if f_high < 0 and high < 10.0:
        high = 10.0
        f_high = _objective_function(high, target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt)

    if f_low * f_high > 0:
        raise ValueError(
            f"Unable to bracket root for target price ${target_price:.2f}. "
            f"f({low:.1%}) = ${f_low:.2f}, f({high:.1%}) = ${f_high:.2f}."
        )

    solver_method = "scipy.optimize.brentq"
    iterations = 0

    if SCIPY_AVAILABLE and not force_fallback:
        root, r_info = brentq(
            _objective_function,
            low,
            high,
            args=(target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt),
            xtol=tol,
            full_output=True
        )
        iterations = r_info.iterations
        implied_growth = root
    else:
        solver_method = "binary_search_fallback"
        implied_growth, iterations = _solve_binary_search(
            target_price, base_fcff, wacc, perpetual_growth, shares, cash, debt,
            low=low, high=high, tol=tol
        )

    # Verification: re-run forward DCF with solved growth rate
    verification_params = DCFParameters(
        base_fcff=base_fcff,
        growth_rate_5yr=implied_growth,
        wacc=wacc,
        perpetual_growth=perpetual_growth,
        shares=shares,
        cash=cash,
        debt=debt
    )
    verification_result = calculate_dcf(verification_params)
    validation_price = verification_result.intrinsic_value_per_share
    pricing_error = abs(validation_price - target_price)

    return ReverseDCFResult(
        target_price=target_price,
        implied_growth_rate=implied_growth,
        target_equity_value=target_equity_value,
        target_enterprise_value=target_enterprise_value,
        base_fcff=base_fcff,
        wacc=wacc,
        perpetual_growth=perpetual_growth,
        iterations=iterations,
        solver_method=solver_method,
        validation_price=validation_price,
        pricing_error=pricing_error
    )
