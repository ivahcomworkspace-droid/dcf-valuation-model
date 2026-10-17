"""
Streamlit Web Dashboard for DCF Valuation Model
Production-ready with dynamic multi-currency support (e.g., ₹ for INR, $ for USD),
resilient error boundaries, empty-state notifications, live yfinance ingestion,
Reverse DCF, and 3-Stage DuPont ROE decomposition.
"""

import os
import sys

# Ensure root workspace is on python path
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from fin_engine.dcf import (
    DCFParameters,
    calculate_dcf,
    generate_sensitivity_matrix,
)
from fin_engine.reverse_dcf import solve_implied_growth
from fin_engine.dupont import calculate_dupont_3stage, decompose_dupont_delta
from services.schemas import get_currency_symbol, CURRENCY_MAP
from services.ingestion import MarketDataService, TickerNotFoundError

# Page Configuration
st.set_page_config(
    page_title="DCF Valuation Model",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Obsidian / Slate aesthetic matching screenshot)
st.markdown("""
<style>
    /* Dark background styling */
    .stApp {
        background-color: #0b0f17;
        color: #f1f5f9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Custom metric card styling */
    .metric-card {
        background-color: #121620;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-card-highlight {
        background: linear-gradient(135deg, rgba(30, 58, 138, 0.35) 0%, rgba(18, 22, 32, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.65);
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 8px 16px -2px rgba(30, 58, 138, 0.3);
    }
    .metric-title {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 6px;
    }
    .metric-title-highlight {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #93c5fd;
        margin-bottom: 6px;
    }
    .metric-val-highlight {
        font-size: 38px;
        font-weight: 800;
        color: #3b82f6;
        line-height: 1.1;
    }
    .metric-val {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        line-height: 1.1;
    }
    .metric-sub {
        font-size: 12px;
        color: #94a3b8;
        margin-top: 6px;
    }
    
    /* Section containers */
    .section-container {
        background-color: #121620;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 20px;
    }
    
    /* Badge styling */
    .cap-badge {
        background-color: #171d2b;
        border: 1px solid #283347;
        border-radius: 8px;
        padding: 6px 14px;
        font-family: monospace;
        font-size: 13px;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session State
if "base_fcff" not in st.session_state:
    st.session_state.base_fcff = 100.0
if "growth_rate" not in st.session_state:
    st.session_state.growth_rate = 8.0
if "wacc" not in st.session_state:
    st.session_state.wacc = 9.0
if "perpetual_growth" not in st.session_state:
    st.session_state.perpetual_growth = 2.5
if "shares" not in st.session_state:
    st.session_state.shares = 20.0
if "cash" not in st.session_state:
    st.session_state.cash = 50.0
if "debt" not in st.session_state:
    st.session_state.debt = 200.0
if "currency_code" not in st.session_state:
    st.session_state.currency_code = "USD"
if "currency_symbol" not in st.session_state:
    st.session_state.currency_symbol = "$"
if "company_name" not in st.session_state:
    st.session_state.company_name = None
if "active_ticker" not in st.session_state:
    st.session_state.active_ticker = None
if "ingestion_notices" not in st.session_state:
    st.session_state.ingestion_notices = []

curr = st.session_state.currency_symbol

# Sidebar: Presets, Capital Structure & Live Ingestion
with st.sidebar:
    st.markdown("### ⚙️ Capital Structure & Presets")
    
    # Presets
    preset = st.selectbox(
        "Load Preset Scenario",
        ["Default (Screenshot Baseline)", "High Growth Tech", "Mature Value / Dividend", "Stress Test"],
        index=0
    )
    if st.button("Apply Preset", use_container_width=True):
        st.session_state.company_name = None
        st.session_state.active_ticker = None
        st.session_state.ingestion_notices = []
        st.session_state.currency_code = "USD"
        st.session_state.currency_symbol = "$"
        if preset == "Default (Screenshot Baseline)":
            st.session_state.base_fcff = 100.0
            st.session_state.growth_rate = 8.0
            st.session_state.wacc = 9.0
            st.session_state.perpetual_growth = 2.5
            st.session_state.shares = 20.0
            st.session_state.cash = 50.0
            st.session_state.debt = 200.0
        elif preset == "High Growth Tech":
            st.session_state.base_fcff = 150.0
            st.session_state.growth_rate = 18.0
            st.session_state.wacc = 10.5
            st.session_state.perpetual_growth = 3.0
            st.session_state.shares = 25.0
            st.session_state.cash = 100.0
            st.session_state.debt = 50.0
        elif preset == "Mature Value / Dividend":
            st.session_state.base_fcff = 120.0
            st.session_state.growth_rate = 3.5
            st.session_state.wacc = 7.5
            st.session_state.perpetual_growth = 2.0
            st.session_state.shares = 15.0
            st.session_state.cash = 40.0
            st.session_state.debt = 120.0
        elif preset == "Stress Test":
            st.session_state.base_fcff = 80.0
            st.session_state.growth_rate = 0.0
            st.session_state.wacc = 11.5
            st.session_state.perpetual_growth = 1.5
            st.session_state.shares = 20.0
            st.session_state.cash = 20.0
            st.session_state.debt = 250.0
        st.rerun()

    st.markdown("---")
    st.markdown("#### Capital Structure")
    st.session_state.shares = st.number_input(
        "Shares Outstanding (M)",
        min_value=0.01,
        value=float(st.session_state.shares),
        step=1.0,
        help="Diluted shares in Millions"
    )
    st.session_state.cash = st.number_input(
        f"Cash & Equivalents ({curr}M)",
        min_value=0.0,
        value=float(st.session_state.cash),
        step=10.0
    )
    st.session_state.debt = st.number_input(
        f"Total Debt ({curr}M)",
        min_value=0.0,
        value=float(st.session_state.debt),
        step=10.0
    )
    net_debt = st.session_state.debt - st.session_state.cash
    st.caption(f"Net Debt (Debt - Cash): **{curr}{net_debt:.1f}M**")

    st.markdown("---")
    st.markdown("#### 💱 Currency Settings")
    currency_options = list(CURRENCY_MAP.keys())
    current_curr_idx = currency_options.index(st.session_state.currency_code) if st.session_state.currency_code in currency_options else 0
    selected_curr = st.selectbox(
        "Reporting Currency",
        currency_options,
        index=current_curr_idx,
        format_func=lambda c: f"{c} ({CURRENCY_MAP.get(c, '')})"
    )
    if selected_curr != st.session_state.currency_code:
        st.session_state.currency_code = selected_curr
        st.session_state.currency_symbol = get_currency_symbol(selected_curr)
        st.rerun()

    st.markdown("---")
    st.markdown("#### 🌐 Live Stock Ingestion (yfinance)")
    ticker_input = st.text_input(
        "Lookup Ticker Symbol",
        value="",
        placeholder="e.g. AAPL, TATASTEEL.NS, TSLA",
        help="For international equities, append the exchange code (e.g. .NS for NSE India, .L for London, .TO for Toronto)."
    )

    fetch_clicked = st.button("Fetch Live Financials", use_container_width=True)
    if fetch_clicked:
        clean_ticker = ticker_input.strip().upper()
        if not clean_ticker:
            st.warning("⚠️ Please enter a stock ticker symbol before fetching.")
        else:
            try:
                with st.spinner(f"Ingesting live market data for {clean_ticker}..."):
                    svc = MarketDataService()
                    bundle = svc.fetch_bundle(clean_ticker, force_refresh=True)
                    snapshot = bundle.market_data
                    financials = bundle.financials

                    st.session_state.active_ticker = snapshot.ticker
                    st.session_state.company_name = snapshot.company_name
                    st.session_state.currency_code = snapshot.currency
                    st.session_state.currency_symbol = snapshot.currency_symbol
                    st.session_state.shares = round(snapshot.diluted_shares, 2)
                    st.session_state.cash = round(snapshot.cash_and_equivalents, 2)
                    st.session_state.debt = round(snapshot.total_debt, 2)

                    # Update FCFF from financials if available
                    inferred_fcf = bundle.infer_base_fcff()
                    if inferred_fcf and inferred_fcf > 0:
                        st.session_state.base_fcff = round(inferred_fcf, 2)

                    st.session_state.ingestion_notices = bundle.warnings
                    st.success(f"Loaded {snapshot.company_name} ({snapshot.ticker}) at {snapshot.currency_symbol}{snapshot.current_price:.2f}/share")
                    st.rerun()

            except TickerNotFoundError as e:
                st.error(f"❌ {str(e)}")
            except Exception as e:
                st.error(f"❌ Failed to retrieve ticker '{clean_ticker}': {str(e)}")

    # Ingestion Notices / Empty State Warnings
    if st.session_state.ingestion_notices:
        with st.expander("ℹ️ Data Feed Notices", expanded=True):
            for notice in st.session_state.ingestion_notices:
                st.markdown(f"- <span style='font-size:12px; color:#e2e8f0;'>{notice}</span>", unsafe_allow_html=True)
            st.caption("You can adjust any parameter manually above.")


# Header Bar
col_header, col_badge = st.columns([1.8, 1.2])
with col_header:
    title_suffix = f" • <span style='color: #60a5fa;'>{st.session_state.company_name} ({st.session_state.active_ticker})</span>" if st.session_state.company_name else ""
    st.markdown(f"<h1 style='margin-bottom: 2px; font-weight: 800;'>DCF Valuation Model{title_suffix}</h1>", unsafe_allow_html=True)
    currency_label = f"Reporting Currency: <b>{st.session_state.currency_code} ({curr})</b>"
    st.markdown(f"<p style='color: #94a3b8; font-size: 13px; margin-top: 0px;'>Discounted Cash Flow & Gordon Growth Valuation &bull; {currency_label}</p>", unsafe_allow_html=True)

with col_badge:
    st.markdown(f"""
    <div style='text-align: right; margin-top: 15px;'>
        <span class='cap-badge'>
            <b>{st.session_state.shares:.1f}M</b> Shares &bull; 
            <b>{curr}{st.session_state.cash:.1f}M</b> Cash &bull; 
            <b>{curr}{st.session_state.debt:.1f}M</b> Debt
        </span>
    </div>
    """, unsafe_allow_html=True)

# Guardrail: WACC must exceed Perpetual Growth
if st.session_state.perpetual_growth >= st.session_state.wacc:
    st.session_state.perpetual_growth = round(st.session_state.wacc - 0.5, 1)

# Valuation Computation via Domain Engine
current_params = DCFParameters(
    base_fcff=float(st.session_state.base_fcff),
    growth_rate_5yr=float(st.session_state.growth_rate) / 100.0,
    wacc=float(st.session_state.wacc) / 100.0,
    perpetual_growth=float(st.session_state.perpetual_growth) / 100.0,
    shares=float(st.session_state.shares),
    cash=float(st.session_state.cash),
    debt=float(st.session_state.debt)
)
dcf_res = calculate_dcf(current_params)


# Main Tabs
tab_dcf, tab_reverse, tab_dupont = st.tabs([
    "📊 DCF Model & Sensitivity",
    "🎯 Reverse DCF (Market Implied)",
    "⚖️ 3-Stage DuPont Analysis"
])

with tab_dcf:
    # 3 Top Metric KPI Cards
    kpi1, kpi2, kpi3 = st.columns(3)
    with kpi1:
        st.markdown(f"""
        <div class='metric-card-highlight'>
            <div class='metric-title-highlight'>INTRINSIC VALUE</div>
            <div class='metric-val-highlight'>{curr}{dcf_res.intrinsic_value_per_share:,.2f}</div>
            <div class='metric-sub'>Per Share</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>ENTERPRISE VALUE</div>
            <div class='metric-val'>{curr}{dcf_res.enterprise_value:,.0f}M</div>
            <div class='metric-sub'><b>{round(dcf_res.terminal_value_pct)}%</b> Terminal Value</div>
        </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>EQUITY VALUE</div>
            <div class='metric-val'>{curr}{dcf_res.equity_value:,.0f}M</div>
            <div class='metric-sub'>EV + Cash - Debt</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    # 5-Year Cash Flow Projections Chart
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px;'>
        <h3 style='margin: 0; font-size: 16px; font-weight: 600;'>5-Year Cash Flow Projections</h3>
        <span style='font-family: monospace; font-size: 12px; color: #94a3b8;'>{curr} Millions (FCFF vs PV)</span>
    </div>
    """, unsafe_allow_html=True)

    years = [f"Year {p.year}" for p in dcf_res.projections]
    fcff_vals = [p.fcff for p in dcf_res.projections]
    pv_vals = [p.present_value for p in dcf_res.projections]

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=years,
        y=fcff_vals,
        name="Projected FCFF",
        marker_color="#93c5fd",
        marker_line_width=0,
        hovertemplate=f"<b>%{{x}} Projected FCFF</b>: {curr}%{{y:.1f}}M<extra></extra>"
    ))
    fig_bars.add_trace(go.Bar(
        x=years,
        y=pv_vals,
        name="Present Value (PV)",
        marker_color="#a5b4fc",
        marker_line_width=0,
        hovertemplate=f"<b>%{{x}} Present Value</b>: {curr}%{{y:.1f}}M<extra></extra>"
    ))

    max_bar_val = max(max(fcff_vals), max(pv_vals)) if fcff_vals else 100.0
    y_upper = max(150, int(np.ceil(max_bar_val / 50.0) * 50))

    fig_bars.update_layout(
        barmode="group",
        bargap=0.35,
        bargroupgap=0.08,
        paper_bgcolor="#121620",
        plot_bgcolor="#121620",
        margin=dict(l=40, r=20, t=20, b=30),
        height=320,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=12)
        ),
        yaxis=dict(
            title="",
            gridcolor="#1e293b",
            gridwidth=1,
            zerolinecolor="#1e293b",
            range=[0, y_upper],
            tickprefix=curr,
            ticksuffix="M",
            tickfont=dict(color="#64748b", family="monospace", size=11)
        ),
        xaxis=dict(
            tickfont=dict(color="#94a3b8", size=12, weight="bold")
        )
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    # Sensitivity Matrix
    st.markdown(f"""
    <div style='display: flex; justify-content: space-between; align-items: baseline; margin-top: 15px; margin-bottom: 8px;'>
        <h3 style='margin: 0; font-size: 16px; font-weight: 600;'>Sensitivity Matrix: Share Price ({curr})</h3>
        <span style='font-family: monospace; font-size: 12px; color: #94a3b8;'>WACC vs Long-term Growth Rate</span>
    </div>
    """, unsafe_allow_html=True)

    matrix_data = generate_sensitivity_matrix(current_params)
    wacc_labels = [f"{w*100:.1f}%" for w in matrix_data["wacc_range"]]
    ltg_labels = [f"{g*100:.1f}%" for g in matrix_data["ltg_range"]]
    z_matrix = matrix_data["matrix"]

    text_matrix = []
    for r_idx, row in enumerate(z_matrix):
        row_text = []
        for c_idx, val in enumerate(row):
            w_val = matrix_data["wacc_range"][r_idx]
            g_val = matrix_data["ltg_range"][c_idx]
            is_active = abs(w_val - current_params.wacc) < 0.0001 and abs(g_val - current_params.perpetual_growth) < 0.0001
            if val is not None:
                if is_active:
                    row_text.append(f"<b>★ {curr}{val:.1f}</b>")
                else:
                    row_text.append(f"{curr}{val:.1f}")
            else:
                row_text.append("N/A")
        text_matrix.append(row_text)

    fig_heat = go.Figure(data=go.Heatmap(
        z=z_matrix,
        x=ltg_labels,
        y=wacc_labels,
        text=text_matrix,
        texttemplate="%{text}",
        textfont=dict(color="#ffffff", size=13, family="monospace"),
        colorscale=[
            [0.0, "#0f172a"],
            [0.3, "#1e3a8a"],
            [0.7, "#1d4ed8"],
            [1.0, "#38bdf8"]
        ],
        showscale=False,
        hovertemplate=f"<b>WACC: %{{y}}</b><br><b>LTG: %{{x}}</b><br>Price: {curr}%{{z:.2f}}<extra></extra>"
    ))

    fig_heat.update_layout(
        paper_bgcolor="#121620",
        plot_bgcolor="#121620",
        margin=dict(l=40, r=20, t=20, b=20),
        height=280,
        yaxis=dict(
            title=dict(text="WACC", font=dict(color="#94a3b8", size=12)),
            autorange="reversed",
            tickfont=dict(color="#cbd5e1", family="monospace", size=11)
        ),
        xaxis=dict(
            title=dict(text="Long-term Growth Rate (LTG)", font=dict(color="#94a3b8", size=12)),
            tickfont=dict(color="#cbd5e1", family="monospace", size=11)
        )
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # 4 Interactive Parameter Sliders
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin: 0 0 14px 0; font-size: 16px; font-weight: 600;'>Interactive Model Parameters</h3>", unsafe_allow_html=True)

    sl_col1, sl_col2 = st.columns(2)
    with sl_col1:
        st.session_state.base_fcff = st.slider(
            f"Base FCFF ({curr}M)",
            min_value=1.0,
            max_value=max(500.0, float(st.session_state.base_fcff) * 2.5),
            value=float(st.session_state.base_fcff),
            step=5.0
        )
        st.session_state.wacc = st.slider(
            "WACC (Discount Rate %)",
            min_value=5.0,
            max_value=20.0,
            value=float(st.session_state.wacc),
            step=0.1
        )

    with sl_col2:
        st.session_state.growth_rate = st.slider(
            "5-Yr Growth Rate (%)",
            min_value=-15.0,
            max_value=35.0,
            value=float(st.session_state.growth_rate),
            step=0.5
        )
        st.session_state.perpetual_growth = st.slider(
            "Perpetual Growth (%)",
            min_value=0.5,
            max_value=6.0,
            value=float(st.session_state.perpetual_growth),
            step=0.1
        )

    # Collapsible Detailed Cash Flow Schedule
    with st.expander("📋 View Detailed Year-by-Year Discounting Schedule"):
        df_sched = pd.DataFrame([
            {
                "Year": f"Year {p.year}",
                "Growth Formula": f"{curr}{current_params.base_fcff:.1f}M × (1 + {current_params.growth_rate_5yr:.1%})^{p.year}",
                "Projected FCFF": f"{curr}{p.fcff:.2f}M",
                "Discount Factor": f"{1/p.discount_factor:.4f}",
                "Present Value (PV)": f"{curr}{p.present_value:.2f}M"
            }
            for p in dcf_res.projections
        ])
        st.table(df_sched)
        st.markdown(f"""
        - **Sum of PV(FCFF)**: `{curr}{dcf_res.sum_pv_fcff:,.2f}M`
        - **Terminal Value (TV₅)**: `{curr}{dcf_res.terminal_value:,.2f}M`
        - **PV of Terminal Value**: `{curr}{dcf_res.pv_terminal_value:,.2f}M`
        - **Enterprise Value**: `{curr}{dcf_res.enterprise_value:,.2f}M`
        - **Equity Value**: `{curr}{dcf_res.equity_value:,.2f}M`
        """)


# TAB 2: Reverse DCF
with tab_reverse:
    st.markdown("### 🎯 Reverse DCF: Solve Market-Implied Growth")
    st.write("Reverse-engineers the 5-year growth rate required to justify any given market share price using Brent's numerical root solver.")

    rev_col1, rev_col2 = st.columns([1, 1.5])
    with rev_col1:
        target_p = st.number_input(
            f"Current Market Share Price ({curr})",
            min_value=0.01,
            max_value=50000.0,
            value=float(round(dcf_res.intrinsic_value_per_share, 2)),
            step=1.0
        )
        run_solver = st.button("Calculate Implied Growth Rate", type="primary", use_container_width=True)

    with rev_col2:
        try:
            rev_sol = solve_implied_growth(
                target_price=target_p,
                base_fcff=current_params.base_fcff,
                wacc=current_params.wacc,
                perpetual_growth=current_params.perpetual_growth,
                shares=current_params.shares,
                cash=current_params.cash,
                debt=current_params.debt
            )

            st.markdown(f"""
            <div class='metric-card-highlight'>
                <div class='metric-title-highlight'>IMPLIED 5-YEAR FCFF GROWTH RATE</div>
                <div class='metric-val-highlight'>{rev_sol.implied_growth_rate:,.2%}</div>
                <div class='metric-sub'>To justify market price of <b>{curr}{target_p:.2f}</b> (Target EV: <b>{curr}{rev_sol.target_enterprise_value:,.0f}M</b>)</div>
            </div>
            """, unsafe_allow_html=True)

            st.caption(f"Solver Method: `{rev_sol.solver_method}` | Iterations: `{rev_sol.iterations}` | Verification Error: `< 1e-4`")

        except Exception as e:
            st.error(f"Solver Error: {e}")

    # Growth Curve vs Target Prices
    st.markdown("#### Market Price Sensitivity Curve")
    sample_prices = np.linspace(max(0.5, target_p * 0.5), target_p * 1.5, 20)
    implied_curve = []
    for sp in sample_prices:
        try:
            sol = solve_implied_growth(
                target_price=sp,
                base_fcff=current_params.base_fcff,
                wacc=current_params.wacc,
                perpetual_growth=current_params.perpetual_growth,
                shares=current_params.shares,
                cash=current_params.cash,
                debt=current_params.debt
            )
            implied_curve.append(sol.implied_growth_rate * 100.0)
        except Exception:
            implied_curve.append(None)

    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(
        x=sample_prices,
        y=implied_curve,
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=6, color="#1d4ed8"),
        name="Implied Growth Rate"
    ))
    fig_curve.add_vline(x=target_p, line_dash="dash", line_color="#ef4444", annotation_text="Target Price")
    fig_curve.update_layout(
        paper_bgcolor="#121620",
        plot_bgcolor="#121620",
        margin=dict(l=40, r=20, t=30, b=30),
        height=280,
        xaxis=dict(title=f"Market Share Price ({curr})", tickprefix=curr, gridcolor="#1e293b", tickfont=dict(color="#cbd5e1")),
        yaxis=dict(title="Implied 5-Yr Growth Rate (%)", ticksuffix="%", gridcolor="#1e293b", tickfont=dict(color="#cbd5e1"))
    )
    st.plotly_chart(fig_curve, use_container_width=True)


# TAB 3: DuPont Analysis
with tab_dupont:
    st.markdown("### ⚖️ 3-Stage DuPont ROE Decomposition")
    st.write("Deconstructs Return on Equity into Operating Profitability (Margin), Asset Utilization (Turnover), and Financial Leverage.")

    dp_col1, dp_col2 = st.columns(2)
    with dp_col1:
        st.markdown("#### Period T-0 (Baseline)")
        ni_0 = st.number_input(f"Net Income T-0 ({curr}M)", value=80.0, step=10.0)
        rev_0 = st.number_input(f"Revenue T-0 ({curr}M)", value=800.0, step=50.0)
        assets_0 = st.number_input(f"Total Assets T-0 ({curr}M)", value=1000.0, step=50.0)
        equity_0 = st.number_input(f"Shareholders' Equity T-0 ({curr}M)", value=400.0, step=50.0)

    with dp_col2:
        st.markdown("#### Period T-1 (Current)")
        ni_1 = st.number_input(f"Net Income T-1 ({curr}M)", value=120.0, step=10.0)
        rev_1 = st.number_input(f"Revenue T-1 ({curr}M)", value=1000.0, step=50.0)
        assets_1 = st.number_input(f"Total Assets T-1 ({curr}M)", value=1200.0, step=50.0)
        equity_1 = st.number_input(f"Shareholders' Equity T-1 ({curr}M)", value=500.0, step=50.0)

    res_dp0 = calculate_dupont_3stage(ni_0, rev_0, assets_0, equity_0)
    res_dp1 = calculate_dupont_3stage(ni_1, rev_1, assets_1, equity_1)
    attr = decompose_dupont_delta(res_dp0, res_dp1)

    st.markdown("---")
    st.markdown(f"#### ROE Comparison: `{res_dp0.roe:.1%}` → `{res_dp1.roe:.1%}` (Δ = **{attr.delta_roe:+.1%}**)")

    dp_kpi1, dp_kpi2, dp_kpi3 = st.columns(3)
    with dp_kpi1:
        st.metric(
            "Net Profit Margin",
            f"{res_dp1.net_profit_margin:.1%}",
            delta=f"{res_dp1.net_profit_margin - res_dp0.net_profit_margin:+.1%}"
        )
        st.caption(f"Margin Attribution: **{attr.margin_effect:+.2%}** ({attr.pct_margin_contribution:.0f}% of Δ)")

    with dp_kpi2:
        st.metric(
            "Asset Turnover",
            f"{res_dp1.asset_turnover:.2f}x",
            delta=f"{res_dp1.asset_turnover - res_dp0.asset_turnover:+.2f}x"
        )
        st.caption(f"Turnover Attribution: **{attr.turnover_effect:+.2%}** ({attr.pct_turnover_contribution:.0f}% of Δ)")

    with dp_kpi3:
        st.metric(
            "Financial Leverage",
            f"{res_dp1.equity_multiplier:.2f}x",
            delta=f"{res_dp1.equity_multiplier - res_dp0.equity_multiplier:+.2f}x"
        )
        st.caption(f"Leverage Attribution: **{attr.leverage_effect:+.2%}** ({attr.pct_leverage_contribution:.0f}% of Δ)")
