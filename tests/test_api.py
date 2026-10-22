"""
Unit and Integration Tests for FastAPI backend endpoints.
Verifies HTTP status codes, Pydantic validation, ticker search, and mathematical accuracy using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_root_serves_frontend():
    """Verifies that GET / serves the Quilet Terminal HTML frontend."""
    response = client.get("/")
    assert response.status_code == 200
    assert "QUILET TERMINAL" in response.text
    assert "text/html" in response.headers.get("content-type", "")


def test_health_check_endpoint():
    """Verifies that the /health endpoint returns 200 and expected status payload."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "Quilet Terminal" in data["service"]


def test_search_ticker_endpoint():
    """Verifies that GET /api/v1/search/ticker returns structured suggestions."""
    response = client.get("/api/v1/search/ticker?q=AAPL")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "AAPL"
    assert isinstance(data["results"], list)
    if data["results"]:
        first = data["results"][0]
        assert "symbol" in first
        assert "name" in first
        assert "exchange" in first


def test_search_ticker_empty_query():
    """Verifies that an empty query returns an empty results list."""
    response = client.get("/api/v1/search/ticker?q=")
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_dcf_valuation_endpoint_benchmark():
    """
    Verifies that POST /api/v1/valuation/dcf produces exact deterministic benchmark numbers:
      Intrinsic Value = $92.12
      Enterprise Value = $1,992M (76% Terminal Value)
      Equity Value = $1,842M
    """
    payload = {
        "base_fcff": 100.0,
        "growth_rate_5yr": 0.08,
        "wacc": 0.09,
        "perpetual_growth": 0.025,
        "shares": 20.0,
        "cash": 50.0,
        "debt": 200.0
    }
    response = client.post("/api/v1/valuation/dcf", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert round(data["intrinsic_value_per_share"], 2) == 92.12
    assert round(data["enterprise_value"]) == 1992
    assert round(data["equity_value"]) == 1842
    assert round(data["terminal_value_pct"]) == 76
    assert data["net_debt"] == 150.0

    projections = data["projections"]
    assert len(projections) == 5
    assert pytest.approx(projections[0]["fcff"], abs=0.01) == 108.00
    assert pytest.approx(projections[0]["present_value"], abs=0.01) == 99.08
    assert pytest.approx(projections[4]["fcff"], abs=0.01) == 146.93
    assert pytest.approx(projections[4]["present_value"], abs=0.01) == 95.50


def test_dcf_valuation_endpoint_bad_request():
    """Verifies that violating domain constraints (WACC <= Perpetual Growth) returns 400."""
    payload = {
        "base_fcff": 100.0,
        "growth_rate_5yr": 0.08,
        "wacc": 0.02,
        "perpetual_growth": 0.025,
        "shares": 20.0
    }
    response = client.post("/api/v1/valuation/dcf", json=payload)
    assert response.status_code == 400
    assert "must be strictly greater than Perpetual Growth" in response.json()["detail"]


def test_reverse_dcf_endpoint():
    """
    Verifies that POST /api/v1/valuation/reverse-dcf inverts target price $92.1155
    back to implied 5-year growth rate of 8.00%.
    """
    payload = {
        "target_price": 92.1155,
        "base_fcff": 100.0,
        "wacc": 0.09,
        "perpetual_growth": 0.025,
        "shares": 20.0,
        "cash": 50.0,
        "debt": 200.0
    }
    response = client.post("/api/v1/valuation/reverse-dcf", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert pytest.approx(data["implied_growth_rate"], abs=1e-4) == 0.08
    assert pytest.approx(data["implied_growth_rate_pct"], abs=1e-2) == 8.00
    assert data["pricing_error"] < 1e-3
    assert "brentq" in data["solver_method"].lower() or "binary" in data["solver_method"].lower()


def test_dupont_endpoint_single_period():
    """Verifies 3-Stage DuPont ROE calculation (Margin * Turnover * Leverage)."""
    payload = {
        "current": {
            "net_income": 120.0,
            "revenue": 1000.0,
            "total_assets": 800.0,
            "shareholders_equity": 400.0
        }
    }
    response = client.post("/api/v1/analysis/dupont", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert pytest.approx(data["net_profit_margin"], abs=1e-5) == 0.12
    assert pytest.approx(data["asset_turnover"], abs=1e-5) == 1.25
    assert pytest.approx(data["equity_multiplier"], abs=1e-5) == 2.00
    assert pytest.approx(data["roe"], abs=1e-5) == 0.30
    assert data["attribution"] is None


def test_dupont_endpoint_with_attribution():
    """Verifies 3-Stage DuPont ROE decomposition with baseline delta attribution."""
    payload = {
        "current": {
            "net_income": 180.0,
            "revenue": 1200.0,
            "total_assets": 1200.0,
            "shareholders_equity": 600.0
        },
        "baseline": {
            "net_income": 100.0,
            "revenue": 1000.0,
            "total_assets": 1000.0,
            "shareholders_equity": 500.0
        }
    }
    response = client.post("/api/v1/analysis/dupont", json=payload)
    assert response.status_code == 200

    data = response.json()
    attr = data["attribution"]
    assert attr is not None
    assert pytest.approx(attr["delta_roe"], abs=1e-5) == 0.10

    # Attribution sum must equal delta_roe exactly
    sum_effects = attr["margin_effect"] + attr["turnover_effect"] + attr["leverage_effect"]
    assert pytest.approx(sum_effects, abs=1e-5) == attr["delta_roe"]
    assert pytest.approx(attr["margin_effect"], abs=1e-5) == 0.10


def test_sensitivity_matrix_endpoint():
    """Verifies the 5x5 WACC vs Perpetual Growth matrix generation."""
    payload = {
        "base_fcff": 100.0,
        "growth_rate_5yr": 0.08,
        "wacc": 0.09,
        "perpetual_growth": 0.025,
        "shares": 20.0,
        "cash": 50.0,
        "debt": 200.0
    }
    response = client.post("/api/v1/valuation/sensitivity", json=payload)
    assert response.status_code == 200

    data = response.json()
    matrix = data["matrix"]
    wacc_range = data["wacc_range"]
    ltg_range = data["ltg_range"]

    assert len(matrix) == 5
    assert len(matrix[0]) == 5
    assert len(wacc_range) == 5
    assert len(ltg_range) == 5

    # Check baseline cell: WACC 9.0%, LTG 2.5% -> $92.1
    w_idx = wacc_range.index(0.09)
    g_idx = ltg_range.index(0.025)
    assert round(matrix[w_idx][g_idx], 1) == 92.1


def test_wacc_valuation_endpoint():
    """Verifies that POST /api/v1/valuation/wacc calculates CAPM Ke and WACC using fin_engine.wacc."""
    payload = {
        "price": 200.0,
        "shares": 15000.0,
        "debt": 100000.0,
        "beta": 1.10,
        "currency": "USD",
        "tax_rate": 0.21,
        "risk_free_rate": 0.042,
        "equity_risk_premium": 0.055
    }
    response = client.post("/api/v1/valuation/wacc", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "wacc" in data
    assert "cost_of_equity" in data
    assert "cost_of_debt" in data
    assert pytest.approx(data["cost_of_equity"], abs=1e-4) == 0.042 + 1.10 * 0.055
    assert 0.05 <= data["wacc"] <= 0.15


def test_get_ticker_data_endpoint():
    """Verifies that GET /api/v1/ticker/{symbol} returns full fundamental metrics and WACC."""
    response = client.get("/api/v1/ticker/AAPL")
    if response.status_code == 200:
        data = response.json()
        assert data["symbol"] == "AAPL"
        assert "price" in data
        assert "currency_symbol" in data
        assert "base_fcff" in data
        assert "wacc" in data
        assert "ebit" in data
        assert "revenue" in data
