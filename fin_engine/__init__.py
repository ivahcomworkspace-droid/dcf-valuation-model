"""
Financial Engine Package (fin_engine)
Pure domain logic for valuation and financial analysis.
Completely isolated from network requests, databases, and I/O.
"""

from fin_engine.dcf import (
    DCFParameters,
    DCFResult,
    CashFlowProjection,
    calculate_dcf,
    generate_sensitivity_matrix,
)
from fin_engine.reverse_dcf import (
    ReverseDCFResult,
    solve_implied_growth,
)
from fin_engine.dupont import (
    DuPont3StageResult,
    calculate_dupont_3stage,
    decompose_dupont_delta,
)
from fin_engine.wacc import (
    calculate_cost_of_equity,
    calculate_wacc,
    calculate_dynamic_wacc,
)

__all__ = [
    "DCFParameters",
    "DCFResult",
    "CashFlowProjection",
    "calculate_dcf",
    "generate_sensitivity_matrix",
    "ReverseDCFResult",
    "solve_implied_growth",
    "DuPont3StageResult",
    "calculate_dupont_3stage",
    "decompose_dupont_delta",
    "calculate_cost_of_equity",
    "calculate_wacc",
    "calculate_dynamic_wacc",
]
