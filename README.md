# DCF Valuation & Financial Analytics Platform

A decoupled, production-grade financial valuation engine and web platform featuring:
1. **Pure Math Core (`fin_engine/`)**: Network-isolated DCF valuation, Reverse DCF root solver, and 3-Stage DuPont attribution.
2. **REST API Backend (`api/main.py`)**: Asynchronous FastAPI endpoints with Pydantic contracts and automated Swagger docs.
3. **Interactive Visual Dashboard (`dashboard/app.py`)**: Streamlit web dashboard with Plotly charts and multi-currency support.
4. **Market Ingestion Service (`services/`)**: Real-time ticker ingestion via `yfinance` with in-memory TTL caching.
5. **Deterministic Pytest Suite (`tests/`)**: 21 unit & integration tests covering domain math, API contracts, and caching.

---

## 🏗️ Decoupled Architectural Blueprint

```
dcf-valuation-model/
├── fin_engine/                  # PURE MATH CORE (Untouched, zero web/network dependencies)
│   ├── __init__.py
│   ├── dcf.py                  # 5-Year DCF, Gordon Growth TV, sensitivity matrix
│   ├── reverse_dcf.py          # SciPy BrentQ root solver for implied growth rate
│   └── dupont.py               # 3-Stage DuPont ROE decomposition & delta attribution
│
├── api/                         # FASTAPI BACKEND SERVICE
│   ├── __init__.py
│   └── main.py                 # Asynchronous REST API endpoints & Pydantic schemas
│
├── services/                    # INGESTION & DATA CONTRACTS
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models & multi-currency mapping
│   └── ingestion.py            # yfinance market data service with TTL cache
│
├── dashboard/                   # STREAMLIT FRONTEND
│   ├── __init__.py
│   └── app.py                  # Streamlit visual dashboard
│
├── tests/                       # DETERMINISTIC TEST SUITE (21 Tests)
│   ├── __init__.py
│   ├── test_api.py             # FastAPI TestClient endpoint verification
│   ├── test_dcf.py             # Deterministic DCF benchmark tests ($92.12)
│   ├── test_reverse_dcf.py     # Reverse DCF convergence & parity tests
│   ├── test_dupont.py          # DuPont math & attribution tests
│   └── test_services.py        # Currency & cache tests
│
├── .streamlit/
│   └── config.toml             # Dark theme tokens & headless server settings
├── .gitignore                   # Excludes venv, pycache, caches, and temp files
├── model.py                     # CLI verification runner
├── requirements.txt             # Verified production dependencies
└── README.md                    # Documentation & API reference
```

---

## ⚡ REST API Endpoints (`api/main.py`)

Run the standalone FastAPI backend server:
```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive API documentation is automatically accessible at:
* Swagger UI: `http://localhost:8000/docs`
* ReDoc: `http://localhost:8000/redoc`

### Endpoints Overview

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Health check endpoint returning API service status |
| `POST` | `/api/v1/valuation/dcf` | 5-Year DCF compounding, present values, EV, and intrinsic share price |
| `POST` | `/api/v1/valuation/reverse-dcf` | Brent's numerical root solver for market-implied growth rate |
| `POST` | `/api/v1/analysis/dupont` | 3-Stage DuPont ROE decomposition with optional delta attribution |
| `POST` | `/api/v1/valuation/sensitivity` | 5x5 WACC vs. Perpetual Growth sensitivity matrix |

---

### Sample API Requests & Responses

#### 1. Forward DCF Valuation (`POST /api/v1/valuation/dcf`)
**Request**:
```json
{
  "base_fcff": 100.0,
  "growth_rate_5yr": 0.08,
  "wacc": 0.09,
  "perpetual_growth": 0.025,
  "shares": 20.0,
  "cash": 50.0,
  "debt": 200.0
}
```
**Response (200 OK)**:
```json
{
  "intrinsic_value_per_share": 92.1155,
  "enterprise_value": 1992.31,
  "equity_value": 1842.31,
  "net_debt": 150.0,
  "terminal_value": 2317.02,
  "pv_terminal_value": 1505.90,
  "terminal_value_pct": 75.59,
  "sum_pv_fcff": 486.41,
  "projections": [
    { "year": 1, "fcff": 108.0, "discount_factor": 1.09, "present_value": 99.08 },
    { "year": 2, "fcff": 116.64, "discount_factor": 1.1881, "present_value": 98.17 },
    { "year": 3, "fcff": 125.97, "discount_factor": 1.2950, "present_value": 97.27 },
    { "year": 4, "fcff": 136.05, "discount_factor": 1.4116, "present_value": 96.38 },
    { "year": 5, "fcff": 146.93, "discount_factor": 1.5386, "present_value": 95.50 }
  ]
}
```

#### 2. Reverse DCF Solver (`POST /api/v1/valuation/reverse-dcf`)
**Request**:
```json
{
  "target_price": 92.1155,
  "base_fcff": 100.0,
  "wacc": 0.09,
  "perpetual_growth": 0.025,
  "shares": 20.0,
  "cash": 50.0,
  "debt": 200.0
}
```
**Response (200 OK)**:
```json
{
  "target_price": 92.1155,
  "implied_growth_rate": 0.08,
  "implied_growth_rate_pct": 8.0,
  "target_equity_value": 1842.31,
  "target_enterprise_value": 1992.31,
  "solver_method": "scipy.optimize.brentq",
  "iterations": 8,
  "validation_price": 92.1155,
  "pricing_error": 0.000000000004
}
```

---

## 🧪 Running Automated Tests

Execute all 21 unit and integration tests across the domain engine, ingestion layer, and FastAPI endpoints:
```powershell
pytest tests/ -v
```
Output:
```text
======================= 21 passed in 11.52s ========================
```

---

## 🚀 Running the Streamlit Frontend

```powershell
streamlit run dashboard/app.py
```
Open `http://localhost:8501` to use the interactive sliders, Plotly charts, and live ticker lookup.
