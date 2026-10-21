"""
FastAPI Backend Application
Exposes the pure financial math engine (fin_engine) as an asynchronous, decoupled REST API.
"""

import os
import sys
from typing import List, Optional

# Ensure project root is in sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from fin_engine.dcf import (
    DCFParameters,
    calculate_dcf,
    generate_sensitivity_matrix,
)
from fin_engine.reverse_dcf import solve_implied_growth
from fin_engine.dupont import calculate_dupont_3stage, decompose_dupont_delta

app = FastAPI(
    title="DCF Valuation & Financial Analytics API",
    version="1.0.0",
    description="Decoupled institutional financial valuation backend API powered by pure fin_engine models."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# Pydantic Request & Response Schemas
# ==========================================

class CashFlowItem(BaseModel):
    year: int
    fcff: float
    discount_factor: float
    present_value: float


class DCFValuationRequest(BaseModel):
    base_fcff: float = Field(..., gt=0, description="Base Free Cash Flow to Firm ($M)")
    growth_rate_5yr: float = Field(..., description="5-Year annual growth rate as decimal (e.g. 0.08 for 8%)")
    wacc: float = Field(..., gt=0, description="Weighted Average Cost of Capital as decimal (e.g. 0.09)")
    perpetual_growth: float = Field(..., gt=0, description="Terminal growth rate as decimal (e.g. 0.025)")
    shares: float = Field(..., gt=0, description="Diluted shares outstanding in Millions")
    cash: float = Field(default=0.0, ge=0, description="Cash & cash equivalents in Millions")
    debt: float = Field(default=0.0, ge=0, description="Total debt in Millions")


class DCFValuationResponse(BaseModel):
    intrinsic_value_per_share: float
    enterprise_value: float
    equity_value: float
    net_debt: float
    terminal_value: float
    pv_terminal_value: float
    terminal_value_pct: float
    sum_pv_fcff: float
    projections: List[CashFlowItem]


class ReverseDCFRequest(BaseModel):
    target_price: float = Field(..., gt=0, description="Target market share price ($)")
    base_fcff: float = Field(..., gt=0, description="Base Free Cash Flow to Firm ($M)")
    wacc: float = Field(..., gt=0, description="WACC as decimal (e.g. 0.09)")
    perpetual_growth: float = Field(..., gt=0, description="Terminal growth rate as decimal (e.g. 0.025)")
    shares: float = Field(..., gt=0, description="Shares outstanding (Millions)")
    cash: float = Field(default=0.0, ge=0, description="Cash & cash equivalents ($M)")
    debt: float = Field(default=0.0, ge=0, description="Total debt ($M)")


class ReverseDCFResponse(BaseModel):
    target_price: float
    implied_growth_rate: float
    implied_growth_rate_pct: float
    target_equity_value: float
    target_enterprise_value: float
    solver_method: str
    iterations: int
    validation_price: float
    pricing_error: float


class DuPontPeriodMetrics(BaseModel):
    net_income: float = Field(..., description="Net income ($M)")
    revenue: float = Field(..., gt=0, description="Total revenue ($M)")
    total_assets: float = Field(..., gt=0, description="Total assets ($M)")
    shareholders_equity: float = Field(..., description="Shareholders equity ($M)")


class DuPontAnalysisRequest(BaseModel):
    current: DuPontPeriodMetrics
    baseline: Optional[DuPontPeriodMetrics] = None


class DuPontAttributionResponse(BaseModel):
    base_roe: float
    current_roe: float
    delta_roe: float
    margin_effect: float
    turnover_effect: float
    leverage_effect: float
    pct_margin_contribution: float
    pct_turnover_contribution: float
    pct_leverage_contribution: float


class DuPontAnalysisResponse(BaseModel):
    net_profit_margin: float
    asset_turnover: float
    equity_multiplier: float
    roe: float
    attribution: Optional[DuPontAttributionResponse] = None


class SensitivityRequest(BaseModel):
    base_fcff: float = Field(..., gt=0, description="Base Free Cash Flow to Firm ($M)")
    growth_rate_5yr: float = Field(..., description="5-Year growth rate as decimal")
    wacc: float = Field(..., gt=0, description="Baseline WACC as decimal")
    perpetual_growth: float = Field(..., gt=0, description="Baseline perpetual growth as decimal")
    shares: float = Field(..., gt=0, description="Shares outstanding in Millions")
    cash: float = Field(default=0.0, ge=0, description="Cash ($M)")
    debt: float = Field(default=0.0, ge=0, description="Debt ($M)")
    wacc_range: Optional[List[float]] = None
    ltg_range: Optional[List[float]] = None


class SensitivityResponse(BaseModel):
    wacc_range: List[float]
    ltg_range: List[float]
    matrix: List[List[Optional[float]]]


# ==========================================
# REST API Endpoints
# ==========================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint confirming API service status."""
    return {
        "status": "healthy",
        "service": "DCF Valuation & Financial Analytics API",
        "version": "1.0.0"
    }


@app.post("/api/v1/valuation/dcf", response_model=DCFValuationResponse, tags=["Valuation"])
async def calculate_dcf_endpoint(req: DCFValuationRequest):
    """
    Computes 5-Year DCF valuation with Gordon Growth Terminal Value.
    Returns explicit cash flows, enterprise value, equity value, and intrinsic share price.
    """
    try:
        params = DCFParameters(
            base_fcff=req.base_fcff,
            growth_rate_5yr=req.growth_rate_5yr,
            wacc=req.wacc,
            perpetual_growth=req.perpetual_growth,
            shares=req.shares,
            cash=req.cash,
            debt=req.debt
        )
        res = calculate_dcf(params)

        projections = [
            CashFlowItem(
                year=p.year,
                fcff=p.fcff,
                discount_factor=p.discount_factor,
                present_value=p.present_value
            )
            for p in res.projections
        ]

        return DCFValuationResponse(
            intrinsic_value_per_share=res.intrinsic_value_per_share,
            enterprise_value=res.enterprise_value,
            equity_value=res.equity_value,
            net_debt=res.net_debt,
            terminal_value=res.terminal_value,
            pv_terminal_value=res.pv_terminal_value,
            terminal_value_pct=res.terminal_value_pct,
            sum_pv_fcff=res.sum_pv_fcff,
            projections=projections
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/valuation/reverse-dcf", response_model=ReverseDCFResponse, tags=["Valuation"])
async def reverse_dcf_endpoint(req: ReverseDCFRequest):
    """
    Calculates the 5-Year annual growth rate implied by a target market share price.
    Uses Brent's root-finding solver (scipy.optimize.brentq) with binary search fallback.
    """
    try:
        sol = solve_implied_growth(
            target_price=req.target_price,
            base_fcff=req.base_fcff,
            wacc=req.wacc,
            perpetual_growth=req.perpetual_growth,
            shares=req.shares,
            cash=req.cash,
            debt=req.debt
        )
        return ReverseDCFResponse(
            target_price=sol.target_price,
            implied_growth_rate=sol.implied_growth_rate,
            implied_growth_rate_pct=sol.implied_growth_rate * 100.0,
            target_equity_value=sol.target_equity_value,
            target_enterprise_value=sol.target_enterprise_value,
            solver_method=sol.solver_method,
            iterations=sol.iterations,
            validation_price=sol.validation_price,
            pricing_error=sol.pricing_error
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/analysis/dupont", response_model=DuPontAnalysisResponse, tags=["Analysis"])
async def dupont_analysis_endpoint(req: DuPontAnalysisRequest):
    """
    Performs 3-Stage DuPont ROE decomposition (Margin * Turnover * Leverage).
    If baseline period metrics are supplied, computes exact sequential delta attribution.
    """
    try:
        curr_res = calculate_dupont_3stage(
            net_income=req.current.net_income,
            revenue=req.current.revenue,
            total_assets=req.current.total_assets,
            shareholders_equity=req.current.shareholders_equity
        )

        attribution_data = None
        if req.baseline is not None:
            base_res = calculate_dupont_3stage(
                net_income=req.baseline.net_income,
                revenue=req.baseline.revenue,
                total_assets=req.baseline.total_assets,
                shareholders_equity=req.baseline.shareholders_equity
            )
            attr = decompose_dupont_delta(base_res, curr_res)
            attribution_data = DuPontAttributionResponse(
                base_roe=attr.base_roe,
                current_roe=attr.current_roe,
                delta_roe=attr.delta_roe,
                margin_effect=attr.margin_effect,
                turnover_effect=attr.turnover_effect,
                leverage_effect=attr.leverage_effect,
                pct_margin_contribution=attr.pct_margin_contribution,
                pct_turnover_contribution=attr.pct_turnover_contribution,
                pct_leverage_contribution=attr.pct_leverage_contribution
            )

        return DuPontAnalysisResponse(
            net_profit_margin=curr_res.net_profit_margin,
            asset_turnover=curr_res.asset_turnover,
            equity_multiplier=curr_res.equity_multiplier,
            roe=curr_res.roe,
            attribution=attribution_data
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/api/v1/valuation/sensitivity", response_model=SensitivityResponse, tags=["Valuation"])
async def sensitivity_matrix_endpoint(req: SensitivityRequest):
    """
    Generates a 2D sensitivity matrix of share prices across WACC and Perpetual Growth rates.
    """
    try:
        params = DCFParameters(
            base_fcff=req.base_fcff,
            growth_rate_5yr=req.growth_rate_5yr,
            wacc=req.wacc,
            perpetual_growth=req.perpetual_growth,
            shares=req.shares,
            cash=req.cash,
            debt=req.debt
        )
        res = generate_sensitivity_matrix(
            params,
            wacc_range=req.wacc_range,
            ltg_range=req.ltg_range
        )
        return SensitivityResponse(
            wacc_range=res["wacc_range"],
            ltg_range=res["ltg_range"],
            matrix=res["matrix"]
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
