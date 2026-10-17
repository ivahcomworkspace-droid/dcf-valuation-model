"""
Unit tests for fin_engine.dcf.
Validates all calculations deterministically against the user's dashboard image benchmarks.
"""

import pytest
from fin_engine.dcf import (
    DCFParameters,
    calculate_dcf,
    generate_sensitivity_matrix,
)


@pytest.fixture
def benchmark_params() -> DCFParameters:
    """Standard baseline parameters matching the user's dashboard image."""
    return DCFParameters(
        base_fcff=100.0,
        growth_rate_5yr=0.08,
        wacc=0.09,
        perpetual_growth=0.025,
        shares=20.0,
        cash=50.0,
        debt=200.0
    )


def test_dcf_projections_and_present_values(benchmark_params):
    """Verifies 5-Year FCFF compounding and discounting."""
    res = calculate_dcf(benchmark_params)

    assert len(res.projections) == 5

    expected_fcff = [108.00, 116.64, 125.97, 136.05, 146.93]
    expected_pv = [99.08, 98.17, 97.27, 96.38, 95.50]

    for i, p in enumerate(res.projections):
        assert pytest.approx(p.fcff, abs=0.01) == expected_fcff[i]
        assert pytest.approx(p.present_value, abs=0.01) == expected_pv[i]

    assert pytest.approx(res.sum_pv_fcff, abs=0.05) == 486.41


def test_dcf_terminal_value_and_kpi_cards(benchmark_params):
    """Verifies Terminal Value, Enterprise Value, Equity Value, and Share Price."""
    res = calculate_dcf(benchmark_params)

    # Terminal Year Cash Flow (Year 6)
    assert pytest.approx(res.terminal_fcff, abs=0.01) == 150.61

    # Undiscounted Terminal Value: 150.606 / (0.09 - 0.025) = 2,317.02
    assert pytest.approx(res.terminal_value, abs=0.1) == 2317.02

    # PV of Terminal Value: 2,317.02 / 1.09^5 = 1,505.90
    assert pytest.approx(res.pv_terminal_value, abs=0.1) == 1505.90

    # Enterprise Value: 486.41 + 1,505.90 = 1,992.31M (Image displays $1,992M)
    assert round(res.enterprise_value) == 1992
    assert pytest.approx(res.enterprise_value, abs=0.1) == 1992.31

    # Terminal Value %: 1,505.90 / 1,992.31 = 75.59% (Image displays 76%)
    assert round(res.terminal_value_pct) == 76

    # Equity Value: 1,992.31 + 50 - 200 = 1,842.31M (Image displays $1,842M)
    assert round(res.equity_value) == 1842
    assert pytest.approx(res.equity_value, abs=0.1) == 1842.31

    # Intrinsic Value Per Share: 1,842.31 / 20 = $92.12 (Image displays $92.12)
    assert round(res.intrinsic_value_per_share, 2) == 92.12


def test_dcf_sensitivity_matrix_benchmark_cells(benchmark_params):
    """Verifies that key cells in the sensitivity matrix exactly match the image table."""
    matrix_data = generate_sensitivity_matrix(benchmark_params)

    wacc_list = matrix_data["wacc_range"]
    ltg_list = matrix_data["ltg_range"]
    table = matrix_data["matrix"]

    # WACC 7.0%, LTG 1.5% -> $114.9
    w_idx = wacc_list.index(0.07)
    g_idx = ltg_list.index(0.015)
    assert round(table[w_idx][g_idx], 1) == 114.9

    # WACC 8.0%, LTG 2.0% -> $102.5
    w_idx = wacc_list.index(0.08)
    g_idx = ltg_list.index(0.020)
    assert round(table[w_idx][g_idx], 1) == 102.5

    # Baseline cell: WACC 9.0%, LTG 2.5% -> $92.1
    w_idx = wacc_list.index(0.09)
    g_idx = ltg_list.index(0.025)
    assert round(table[w_idx][g_idx], 1) == 92.1

    # WACC 10.0%, LTG 3.0% -> $83.3
    w_idx = wacc_list.index(0.10)
    g_idx = ltg_list.index(0.030)
    assert round(table[w_idx][g_idx], 1) == 83.3

    # WACC 11.0%, LTG 3.5% -> $75.7
    w_idx = wacc_list.index(0.11)
    g_idx = ltg_list.index(0.035)
    assert round(table[w_idx][g_idx], 1) == 75.7


def test_dcf_validation_constraints():
    """Verifies domain constraints and error handling."""
    # WACC <= Perpetual Growth
    with pytest.raises(ValueError, match="WACC .* must be strictly greater than Perpetual Growth"):
        calculate_dcf(DCFParameters(
            base_fcff=100.0,
            growth_rate_5yr=0.08,
            wacc=0.025,
            perpetual_growth=0.025,
            shares=20.0
        ))

    # Shares <= 0
    with pytest.raises(ValueError, match="Shares outstanding must be positive"):
        calculate_dcf(DCFParameters(
            base_fcff=100.0,
            growth_rate_5yr=0.08,
            wacc=0.09,
            perpetual_growth=0.025,
            shares=0.0
        ))
