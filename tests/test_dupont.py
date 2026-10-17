"""
Unit tests for fin_engine.dupont.
Validates 3-Stage DuPont decomposition and exact path delta attribution.
"""

import pytest
from fin_engine.dupont import calculate_dupont_3stage, decompose_dupont_delta


def test_dupont_3stage_decomposition():
    """
    Given:
      Net Income = $120M
      Revenue = $1,000M
      Total Assets = $800M
      Equity = $400M
    Then:
      Margin = 120 / 1000 = 12.0%
      Turnover = 1000 / 800 = 1.25x
      Leverage = 800 / 400 = 2.0x
      ROE = 12% * 1.25 * 2.0 = 30.0% (also 120 / 400 = 30.0%)
    """
    res = calculate_dupont_3stage(
        net_income=120.0,
        revenue=1000.0,
        total_assets=800.0,
        shareholders_equity=400.0
    )

    assert pytest.approx(res.net_profit_margin, abs=1e-5) == 0.12
    assert pytest.approx(res.asset_turnover, abs=1e-5) == 1.25
    assert pytest.approx(res.equity_multiplier, abs=1e-5) == 2.00
    assert pytest.approx(res.roe, abs=1e-5) == 0.30


def test_dupont_delta_attribution():
    """
    Period 0:
      Net Income = $100M, Revenue = $1000M, Assets = $1000M, Equity = $500M
      ROE_0 = 10% * 1.0 * 2.0 = 20.0%
    Period 1:
      Net Income = $180M, Revenue = $1200M, Assets = $1200M, Equity = $600M
      Margin: 180 / 1200 = 15.0%
      Turnover: 1200 / 1200 = 1.0x
      Leverage: 1200 / 600 = 2.0x
      ROE_1 = 15% * 1.0 * 2.0 = 30.0%
      Delta ROE = +10.0% (all driven by margin expansion)
    """
    t0 = calculate_dupont_3stage(100.0, 1000.0, 1000.0, 500.0)
    t1 = calculate_dupont_3stage(180.0, 1200.0, 1200.0, 600.0)

    attr = decompose_dupont_delta(t0, t1)

    assert pytest.approx(attr.delta_roe, abs=1e-5) == 0.10
    # Sum of effects must equal delta_roe exactly
    sum_effects = attr.margin_effect + attr.turnover_effect + attr.leverage_effect
    assert pytest.approx(sum_effects, abs=1e-5) == attr.delta_roe
    assert pytest.approx(attr.margin_effect, abs=1e-5) == 0.10
    assert pytest.approx(attr.turnover_effect, abs=1e-5) == 0.0
    assert pytest.approx(attr.leverage_effect, abs=1e-5) == 0.0
