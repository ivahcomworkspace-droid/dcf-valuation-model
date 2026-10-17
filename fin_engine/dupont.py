"""
3-Stage DuPont Analysis Engine
Decomposes Return on Equity (ROE) into Net Profit Margin, Asset Turnover, and Financial Leverage,
including delta attribution between periods or against a benchmark.
Completely isolated from network calls.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class DuPont3StageResult:
    """Results of a single-period 3-Stage DuPont decomposition."""
    net_income: float            # $M
    revenue: float               # $M
    total_assets: float          # $M
    shareholders_equity: float   # $M
    
    # 3 Core Components
    net_profit_margin: float     # Net Income / Revenue
    asset_turnover: float        # Revenue / Total Assets
    equity_multiplier: float     # Total Assets / Shareholders' Equity (Financial Leverage)
    
    # Return on Equity
    roe: float                   # Net Income / Equity = Margin * Turnover * Leverage


@dataclass(frozen=True)
class DuPontAttributionResult:
    """Attribution of delta in ROE between base period (t0) and current period (t1)."""
    base_roe: float
    current_roe: float
    delta_roe: float
    margin_effect: float         # Contribution from change in Net Profit Margin
    turnover_effect: float       # Contribution from change in Asset Turnover
    leverage_effect: float       # Contribution from change in Financial Leverage
    
    # Percentage shares of the total change
    pct_margin_contribution: float
    pct_turnover_contribution: float
    pct_leverage_contribution: float


def calculate_dupont_3stage(
    net_income: float,
    revenue: float,
    total_assets: float,
    shareholders_equity: float
) -> DuPont3StageResult:
    """
    Computes 3-Stage DuPont decomposition:
        ROE = Net Profit Margin * Asset Turnover * Equity Multiplier
    """
    if revenue <= 0:
        raise ValueError(f"Revenue must be strictly positive, got {revenue}")
    if total_assets <= 0:
        raise ValueError(f"Total Assets must be strictly positive, got {total_assets}")
    if shareholders_equity == 0:
        raise ValueError("Shareholders' Equity cannot be zero.")

    margin = net_income / revenue
    turnover = revenue / total_assets
    leverage = total_assets / shareholders_equity
    roe = margin * turnover * leverage

    return DuPont3StageResult(
        net_income=net_income,
        revenue=revenue,
        total_assets=total_assets,
        shareholders_equity=shareholders_equity,
        net_profit_margin=margin,
        asset_turnover=turnover,
        equity_multiplier=leverage,
        roe=roe
    )


def decompose_dupont_delta(
    base: DuPont3StageResult,
    current: DuPont3StageResult
) -> DuPontAttributionResult:
    """
    Decomposes the delta in ROE between two periods into margin, turnover, and leverage effects.
    Uses sequential exact path attribution such that:
        delta_roe = margin_effect + turnover_effect + leverage_effect (no residual).
    """
    delta_roe = current.roe - base.roe

    # Step-by-step substitution:
    # 1. Change margin holding turnover and leverage at base:
    margin_effect = (current.net_profit_margin - base.net_profit_margin) * base.asset_turnover * base.equity_multiplier
    # 2. Change turnover with new margin, holding leverage at base:
    turnover_effect = current.net_profit_margin * (current.asset_turnover - base.asset_turnover) * base.equity_multiplier
    # 3. Change leverage with new margin and new turnover:
    leverage_effect = current.net_profit_margin * current.asset_turnover * (current.equity_multiplier - base.equity_multiplier)

    if abs(delta_roe) > 1e-9:
        pct_margin = (margin_effect / delta_roe) * 100.0
        pct_turnover = (turnover_effect / delta_roe) * 100.0
        pct_leverage = (leverage_effect / delta_roe) * 100.0
    else:
        pct_margin = 0.0
        pct_turnover = 0.0
        pct_leverage = 0.0

    return DuPontAttributionResult(
        base_roe=base.roe,
        current_roe=current.roe,
        delta_roe=delta_roe,
        margin_effect=margin_effect,
        turnover_effect=turnover_effect,
        leverage_effect=leverage_effect,
        pct_margin_contribution=pct_margin,
        pct_turnover_contribution=pct_turnover,
        pct_leverage_contribution=pct_leverage
    )
