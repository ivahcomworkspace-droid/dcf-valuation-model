# DCF Valuation & Financial Analysis Engine

A production-ready financial valuation dashboard and analytics suite built for deployment to **Streamlit Community Cloud** and **GitHub**.

---

## 🌟 Highlights & Features

1. **Interactive DCF Valuation Dashboard**:
   * Replicates the exact visual aesthetic from the reference model.
   * **Dynamic Multi-Currency Detection**: Auto-detects reporting currencies (e.g., `₹` for INR stocks like `TATASTEEL.NS`, `$` for US equities, `€` for European equities, `£` for UK equities).
   * **KPI Summary Cards**: Intrinsic Value per share, Enterprise Value, and Equity Value.
   * **5-Year Cash Flow Projections Chart**: Plotly grouped chart for Projected FCFF vs. Present Value (PV).
   * **Sensitivity Matrix**: 2D Heatmap across WACC (7%–11%) and Perpetual Growth (1.5%–3.5%) with active-scenario highlighting.
   * **Interactive Sliders**: Real-time adjustment of Base FCFF, Growth Rate, WACC, and Perpetual Growth.

2. **Reverse DCF Analysis**:
   * Uses Brent's numerical root solver (`scipy.optimize.brentq`) with binary search fallback to invert market prices into **implied 5-year growth rates**.
   * Interactive market price sensitivity curve.

3. **3-Stage DuPont Decomposition**:
   * Deconstructs ROE into Net Profit Margin $\times$ Asset Turnover $\times$ Equity Multiplier.
   * Exact path sequential delta attribution ($\Delta \text{ROE}$).

4. **Live Market Ingestion (`services/`)**:
   * Ingests real-time financial data via `yfinance` with Pydantic validation and in-memory TTL caching.
   * Graceful error boundaries and missing-data notices for international tickers.

---

## 📁 Repository Structure

```
dcf-valuation-model/
├── .streamlit/
│   └── config.toml             # Dark theme tokens, headless flags, and server settings
├── dashboard/
│   ├── __init__.py
│   └── app.py                  # Main Streamlit application
├── fin_engine/                  # Pure domain logic (isolated from network/IO)
│   ├── __init__.py
│   ├── dcf.py                  # Forward DCF & 2D sensitivity matrix
│   ├── reverse_dcf.py          # Reverse DCF BrentQ solver
│   └── dupont.py               # 3-Stage DuPont ROE engine
├── services/                    # Ingestion & external API layer
│   ├── __init__.py
│   ├── schemas.py              # Pydantic models & currency mapping
│   └── ingestion.py            # yfinance service with TTL cache
├── tests/                       # Deterministic test suite (pytest)
│   ├── __init__.py
│   ├── test_dcf.py             # Baseline math tests ($92.12 benchmark)
│   ├── test_reverse_dcf.py     # Reverse DCF convergence tests
│   ├── test_dupont.py          # DuPont attribution tests
│   └── test_services.py        # Currency & cache tests
├── .gitignore                   # Excludes virtual environments, caches, and temp files
├── model.py                     # CLI verification runner
├── requirements.txt             # Verified production dependencies
└── README.md                    # Documentation & deployment guide
```

---

## 🚀 Running Locally

1. **Clone the repository**:
   ```bash
   git clone https://github.com/<your-username>/dcf-valuation-model.git
   cd dcf-valuation-model
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Automated Tests**:
   ```bash
   pytest tests/ -v
   ```

4. **Launch the Web Dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```
   Open `http://localhost:8501` in your browser.

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"**.
4. Select your GitHub repository (`dcf-valuation-model`), branch (`main`), and set **Main file path** to:
   ```text
   dashboard/app.py
   ```
5. Click **"Deploy!"** — Streamlit Cloud will install dependencies from `requirements.txt` and configure theme settings from `.streamlit/config.toml`.
