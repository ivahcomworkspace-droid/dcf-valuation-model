"""
Discounted Cash Flow (DCF) Domain Model
Pure, network-isolated implementation of DCF and Gordon Growth Terminal Value.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass(frozen=True)
class CashFlowProjection:
    """Represents a single projected year's cash flow and present value."""
    year: int
    fcff: float
    discount_factor: float
    present_value: float


@dataclass(frozen=True)
class DCFParameters:
    """Inputs required for 5-Year DCF valuation with Gordon Growth Terminal Value."""
    base_fcff: float          # Starting Free Cash Flow to Firm ($M)
    growth_rate_5yr: float    # 5-year annual growth rate (decimal, e.g. 0.08 for 8.0%)
    wacc: float               # Weighted Average Cost of Capital (decimal, e.g. 0.09 for 9.0%)
    perpetual_growth: float   # Terminal growth rate (decimal, e.g. 0.025 for 2.5%)
    shares: float             # Diluted shares outstanding (Millions, e.g. 20.0)
    cash: float = 0.0         # Cash and cash equivalents ($M)
    debt: float = 0.0         # Total debt ($M)

    def validate(self) -> None:
        """Validate financial parameters against theoretical domain constraints."""
        if self.shares <= 0:
            raise ValueError(f"Shares outstanding must be positive, got {self.shares}")
        if self.wacc <= self.perpetual_growth:
            raise ValueError(
                f"WACC ({self.wacc:.2%}) must be strictly greater than Perpetual Growth "
                f"({self.perpetual_growth:.2%}) for the Gordon Growth Model to converge."
            )
        if self.wacc <= 0:
            raise ValueError(f"WACC must be strictly positive, got {self.wacc:.2%}")
        if self.base_fcff < 0:
            # We allow negative FCFF in distress models, but warn/flag if needed
            pass


@dataclass(frozen=True)
class DCFResult:
    """Output metrics of a completed DCF valuation run."""
    parameters: DCFParameters
    projections: List[CashFlowProjection]
    sum_pv_fcff: float
    terminal_fcff: float
    terminal_value: float
    pv_terminal_value: float
    terminal_value_pct: float
    enterprise_value: float
    net_debt: float
    equity_value: float
    intrinsic_value_per_share: float


def calculate_dcf(params: DCFParameters) -> DCFResult:
    """
    Executes a 5-Year Discounted Cash Flow valuation.
    
    Formulas:
        FCFF_t = FCFF_0 * (1 + g_1)^t
        PV(FCFF_t) = FCFF_t / (1 + WACC)^t
        FCFF_6 = FCFF_5 * (1 + g_2)
        TV_5 = FCFF_6 / (WACC - g_2)
        PV(TV) = TV_5 / (1 + WACC)^5
        Enterprise Value = Sum(PV_FCFF) + PV(TV)
        Equity Value = Enterprise Value + Cash - Debt
        Intrinsic Value Per Share = Equity Value / Shares
    """
    params.validate()

    projections: List[CashFlowProjection] = []
    sum_pv_fcff = 0.0

    for year in range(1, 6):
        fcff = params.base_fcff * ((1.0 + params.growth_rate_5yr) ** year)
        discount_factor = (1.0 + params.wacc) ** year
        pv = fcff / discount_factor
        sum_pv_fcff += pv
        projections.append(CashFlowProjection(
            year=year,
            fcff=fcff,
            discount_factor=discount_factor,
            present_value=pv
        ))

    # Terminal Value via Gordon Growth Model
    terminal_fcff = projections[-1].fcff * (1.0 + params.perpetual_growth)
    terminal_value = terminal_fcff / (params.wacc - params.perpetual_growth)
    pv_terminal_value = terminal_value / ((1.0 + params.wacc) ** 5)

    enterprise_value = sum_pv_fcff + pv_terminal_value
    terminal_value_pct = (pv_terminal_value / enterprise_value * 100.0) if enterprise_value > 0 else 0.0

    net_debt = params.debt - params.cash
    equity_value = enterprise_value - net_debt
    intrinsic_value_per_share = equity_value / params.shares

    return DCFResult(
        parameters=params,
        projections=projections,
        sum_pv_fcff=sum_pv_fcff,
        terminal_fcff=terminal_fcff,
        terminal_value=terminal_value,
        pv_terminal_value=pv_terminal_value,
        terminal_value_pct=terminal_value_pct,
        enterprise_value=enterprise_value,
        net_debt=net_debt,
        equity_value=equity_value,
        intrinsic_value_per_share=intrinsic_value_per_share,
    )


def generate_sensitivity_matrix(
    params: DCFParameters,
    wacc_range: Optional[List[float]] = None,
    ltg_range: Optional[List[float]] = None
) -> Dict[str, any]:
    """
    Generates a 2D sensitivity matrix of share prices across WACC and Long-Term Growth rates.
    Default grid matches the screenshot:
      WACC: [7.0%, 8.0%, 9.0%, 10.0%, 11.0%]
      LTG:  [1.5%, 2.0%, 2.5%, 3.0%, 3.5%]
    """
    if wacc_range is None:
        wacc_range = [0.07, 0.08, 0.09, 0.10, 0.11]
    if ltg_range is None:
        ltg_range = [0.015, 0.020, 0.025, 0.030, 0.035]

    matrix: List[List[Optional[float]]] = []

    for w in wacc_range:
        row: List[Optional[float]] = []
        for g in ltg_range:
            if w <= g:
                row.append(None)
                continue
            scenario_params = DCFParameters(
                base_fcff=params.base_fcff,
                growth_rate_5yr=params.growth_rate_5yr,
                wacc=w,
                perpetual_growth=g,
                shares=params.shares,
                cash=params.cash,
                debt=params.debt
            )
            res = calculate_dcf(scenario_params)
            row.append(res.intrinsic_value_per_share)
        matrix.append(row)

    return {
        "wacc_range": wacc_range,
        "ltg_range": ltg_range,
        "matrix": matrix,
    }
