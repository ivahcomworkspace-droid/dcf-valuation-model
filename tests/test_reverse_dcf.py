"""
Unit tests for fin_engine.reverse_dcf.
Validates the numerical solver, round-trip convergence, and boundary conditions.
"""

import pytest
from fin_engine.dcf import DCFParameters, calculate_dcf
from fin_engine.reverse_dcf import solve_implied_growth


def test_reverse_dcf_roundtrip_convergence():
    """
    Roundtrip test:
    Given g = 8.0%, Forward DCF computes price P = $92.1155.
    Reverse DCF must invert P back to g = 8.000% within tolerance.
    """
    baseline_params = DCFParameters(
        base_fcff=100.0,
        growth_rate_5yr=0.08,
        wacc=0.09,
        perpetual_growth=0.025,
        shares=20.0,
        cash=50.0,
        debt=200.0
    )
    forward_result = calculate_dcf(baseline_params)
    target_price = forward_result.intrinsic_value_per_share

    # Invert target price
    reverse_res = solve_implied_growth(
        target_price=target_price,
        base_fcff=100.0,
        wacc=0.09,
        perpetual_growth=0.025,
        shares=20.0,
        cash=50.0,
        debt=200.0
    )

    # Implied growth must equal 8.000%
    assert pytest.approx(reverse_res.implied_growth_rate, abs=1e-5) == 0.08
    assert reverse_res.pricing_error < 1e-4
    assert pytest.approx(reverse_res.validation_price, abs=1e-4) == target_price


def test_brentq_vs_binary_search_parity():
    """Verifies that SciPy BrentQ and the binary search fallback produce identical results."""
    target_price = 110.50

    res_brentq = solve_implied_growth(
        target_price=target_price,
        base_fcff=100.0,
        wacc=0.09,
        perpetual_growth=0.025,
        shares=20.0,
        cash=50.0,
        debt=200.0,
        force_fallback=False
    )

    res_binary = solve_implied_growth(
        target_price=target_price,
        base_fcff=100.0,
        wacc=0.09,
        perpetual_growth=0.025,
        shares=20.0,
        cash=50.0,
        debt=200.0,
        force_fallback=True
    )

    assert pytest.approx(res_brentq.implied_growth_rate, abs=1e-4) == res_binary.implied_growth_rate
    assert pytest.approx(res_brentq.validation_price, abs=1e-3) == target_price
    assert pytest.approx(res_binary.validation_price, abs=1e-3) == target_price


def test_reverse_dcf_monotonicity():
    """Verifies that higher market prices strictly imply higher growth expectations."""
    prices = [60.0, 75.0, 92.12, 110.0, 140.0]
    implied_growths = []

    for p in prices:
        r = solve_implied_growth(
            target_price=p,
            base_fcff=100.0,
            wacc=0.09,
            perpetual_growth=0.025,
            shares=20.0,
            cash=50.0,
            debt=200.0
        )
        implied_growths.append(r.implied_growth_rate)

    # Ensure strictly increasing
    for i in range(len(implied_growths) - 1):
        assert implied_growths[i] < implied_growths[i + 1]


def test_reverse_dcf_error_handling():
    """Verifies validation on non-positive share prices or impossible valuations."""
    with pytest.raises(ValueError, match="strictly positive"):
        solve_implied_growth(
            target_price=-10.0,
            base_fcff=100.0,
            wacc=0.09,
            perpetual_growth=0.025,
            shares=20.0
        )
