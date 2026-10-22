"""
Market Data Ingestion Service
Fetches live market data and financial statements using yfinance with in-memory TTL caching,
dynamic currency detection, and comprehensive data completeness validation.
"""

from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
import logging

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

from services.schemas import (
    MarketDataSnapshot,
    FinancialStatementItems,
    DCFIngestionBundle,
    get_currency_symbol,
)

logger = logging.getLogger(__name__)


class TickerNotFoundError(Exception):
    """Raised when the specified stock ticker symbol does not exist or has no trading history."""
    pass


class MarketDataService:
    """
    Service responsible for fetching and caching market data and financial statement metrics.
    Caches snapshots in-memory with a configurable Time-To-Live (TTL).
    """

    def __init__(self, ttl_seconds: int = 300):
        self.ttl_seconds = ttl_seconds
        # Cache storage: ticker -> (cache_timestamp, MarketDataSnapshot, Optional[FinancialStatementItems])
        self._cache: Dict[str, Tuple[datetime, MarketDataSnapshot, Optional[FinancialStatementItems]]] = {}

    def clear_cache(self) -> None:
        """Clears all cached data."""
        self._cache.clear()

    def _is_cache_valid(self, cached_time: datetime) -> bool:
        now = datetime.now(timezone.utc)
        elapsed = (now - cached_time).total_seconds()
        return elapsed < self.ttl_seconds

    def fetch_market_data(self, ticker: str, force_refresh: bool = False) -> MarketDataSnapshot:
        """
        Fetches live market data (price, currency, shares, cash, debt, beta, interest expense).
        Returns validated MarketDataSnapshot.
        """
        symbol = ticker.strip().upper()
        if not symbol:
            raise ValueError("Ticker symbol cannot be empty.")

        if not force_refresh and symbol in self._cache:
            cached_time, snapshot, _ = self._cache[symbol]
            if self._is_cache_valid(cached_time):
                return snapshot

        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance library is not available. Please install yfinance.")

        try:
            yf_ticker = yf.Ticker(symbol)
            info = yf_ticker.info or {}

            # Price detection with fallback cascade
            price = (
                info.get("currentPrice")
                or info.get("regularMarketPrice")
                or info.get("ask")
                or (getattr(yf_ticker, "fast_info", None) and getattr(yf_ticker.fast_info, "last_price", None))
            )
            if not price:
                hist = yf_ticker.history(period="5d")
                if not hist.empty and "Close" in hist:
                    price = float(hist["Close"].iloc[-1])

            if not price or price <= 0:
                raise TickerNotFoundError(
                    f"Quote not found for '{symbol}'. Please ensure the ticker symbol is valid "
                    f"and includes the exchange suffix if outside the US (e.g., TATASTEEL.NS for NSE India, "
                    f"AZN.L for London, SHOP.TO for Toronto)."
                )

            currency = info.get("currency") or "USD"
            company_name = info.get("shortName") or info.get("longName") or symbol

            warnings: List[str] = []

            # Diluted Shares (converted to Millions)
            raw_shares = info.get("sharesOutstanding") or info.get("impliedSharesOutstanding")
            if raw_shares:
                diluted_shares = float(raw_shares) / 1e6
            else:
                diluted_shares = 100.0
                warnings.append("Shares Outstanding was not disclosed; defaulted to 100.0M.")

            # Cash and Equivalents (converted to Millions)
            raw_cash = info.get("totalCash")
            if raw_cash is not None:
                cash = float(raw_cash) / 1e6
            else:
                cash = 0.0
                warnings.append("Cash balance was not disclosed; defaulted to 0.")

            # Total Debt (converted to Millions)
            raw_debt = info.get("totalDebt")
            if raw_debt is not None:
                debt = float(raw_debt) / 1e6
            else:
                debt = 0.0
                warnings.append("Total Debt was not disclosed; defaulted to 0.")

            # Beta
            beta = info.get("beta")
            if beta is not None:
                beta = float(beta)

            # Market Cap (Millions)
            raw_mcap = info.get("marketCap")
            market_cap = float(raw_mcap) / 1e6 if raw_mcap else None

            # Interest Expense (Millions)
            raw_interest = info.get("interestExpense")
            interest_expense = float(raw_interest) / 1e6 if raw_interest is not None else None

            snapshot = MarketDataSnapshot(
                ticker=symbol,
                company_name=company_name,
                current_price=float(price),
                currency=currency,
                diluted_shares=diluted_shares,
                market_cap=market_cap,
                cash_and_equivalents=cash,
                total_debt=debt,
                beta=beta,
                interest_expense=interest_expense,
                timestamp=datetime.now(timezone.utc),
                warnings=warnings,
            )

            # Update cache entry
            existing_financials = self._cache.get(symbol, (None, None, None))[2]
            self._cache[symbol] = (datetime.now(timezone.utc), snapshot, existing_financials)
            return snapshot

        except TickerNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            raise TickerNotFoundError(f"Could not retrieve market data for '{symbol}': {str(e)}")

    def fetch_financial_statements(self, ticker: str, force_refresh: bool = False) -> Optional[FinancialStatementItems]:
        """
        Fetches income statement, cash flow, and balance sheet items for DuPont & DCF.
        """
        symbol = ticker.strip().upper()

        if not force_refresh and symbol in self._cache:
            cached_time, _, cached_fin = self._cache[symbol]
            if cached_fin is not None and self._is_cache_valid(cached_time):
                return cached_fin

        if not YFINANCE_AVAILABLE:
            raise RuntimeError("yfinance library is not available.")

        try:
            yf_ticker = yf.Ticker(symbol)
            info = yf_ticker.info or {}

            raw_revenue = info.get("totalRevenue")
            raw_net_income = info.get("netIncomeToCommon") or info.get("netIncome")
            currency = info.get("currency") or "USD"

            inc = getattr(yf_ticker, "income_stmt", None)
            cf = getattr(yf_ticker, "cashflow", None)
            bs = getattr(yf_ticker, "balance_sheet", None)

            def _get_val(df, keys):
                if df is not None and not df.empty:
                    for k in keys:
                        if k in df.index:
                            try:
                                v = df.loc[k].dropna().iloc[0]
                                return float(v)
                            except Exception:
                                pass
                return None

            # Balance sheet items
            raw_assets = _get_val(bs, ["Total Assets"])
            raw_equity = _get_val(bs, ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"])

            # Cash flow items
            raw_ocf = info.get("operatingCashflow") or _get_val(cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"])
            raw_fcf = info.get("freeCashflow") or _get_val(cf, ["Free Cash Flow"])

            # Fundamental FCFF items: EBIT, Tax Rate, D&A, CapEx, ΔNWC
            raw_ebit = _get_val(inc, ["EBIT", "Operating Income", "Normalized EBITDA"]) or info.get("operatingIncome")
            raw_dna = _get_val(cf, ["Depreciation & Amortization", "Depreciation And Amortization", "Reconciled Depreciation"])
            if raw_dna is None:
                raw_dna = _get_val(inc, ["Reconciled Depreciation", "Depreciation & Amortization"])

            raw_capex = _get_val(cf, ["Capital Expenditure", "Capital Expenditures"])
            if raw_capex is not None:
                raw_capex = abs(raw_capex)
            elif info.get("capitalExpenditures"):
                raw_capex = abs(float(info.get("capitalExpenditures")))

            raw_dnwc = _get_val(cf, ["Change In Working Capital", "Changes In Working Capital"]) or 0.0

            # Tax Rate resolution
            raw_tax_rate = _get_val(inc, ["Tax Rate For Calcs"])
            if raw_tax_rate is None or raw_tax_rate <= 0 or raw_tax_rate >= 0.70:
                raw_tax_prov = _get_val(inc, ["Tax Provision"]) or 0.0
                raw_pretax = _get_val(inc, ["Pretax Income"]) or 0.0
                if raw_pretax > 0 and raw_tax_prov > 0:
                    raw_tax_rate = raw_tax_prov / raw_pretax
                else:
                    raw_tax_rate = 0.25 if currency.upper() == "INR" else 0.21

            # Formula-derived FCFF = EBIT*(1-t) + D&A - CapEx - ΔNWC
            computed_fcff = None
            if raw_ebit is not None and raw_dna is not None and raw_capex is not None:
                fcff_val = (raw_ebit * (1.0 - raw_tax_rate) + raw_dna - raw_capex - raw_dnwc) / 1e6
                if fcff_val > 0:
                    computed_fcff = fcff_val

            if not raw_revenue or raw_revenue <= 0:
                return None

            financials = FinancialStatementItems(
                ticker=symbol,
                currency=currency,
                revenue=float(raw_revenue) / 1e6,
                net_income=float(raw_net_income) / 1e6 if raw_net_income else 0.0,
                total_assets=float(raw_assets) / 1e6 if raw_assets else float(raw_revenue) / 1e6,
                total_equity=float(raw_equity) / 1e6 if raw_equity else (float(raw_assets or raw_revenue) * 0.5) / 1e6,
                operating_cash_flow=float(raw_ocf) / 1e6 if raw_ocf else None,
                capital_expenditures=float(raw_capex) / 1e6 if raw_capex else None,
                free_cash_flow=float(raw_fcf) / 1e6 if raw_fcf else None,
                ebit=float(raw_ebit) / 1e6 if raw_ebit else None,
                effective_tax_rate=float(raw_tax_rate),
                depreciation_amortization=float(raw_dna) / 1e6 if raw_dna else None,
                change_in_working_capital=float(raw_dnwc) / 1e6 if raw_dnwc else None,
                computed_fcff=computed_fcff,
            )

            # Store in cache
            cached_entry = self._cache.get(symbol)
            snapshot = cached_entry[1] if cached_entry else None
            if snapshot is None:
                snapshot = self.fetch_market_data(symbol, force_refresh=force_refresh)
            self._cache[symbol] = (datetime.now(timezone.utc), snapshot, financials)
            return financials

        except Exception as e:
            logger.error(f"Failed to fetch financial statements for {symbol}: {e}")
            return None

    def fetch_bundle(self, ticker: str, force_refresh: bool = False) -> DCFIngestionBundle:
        """Fetches both market data and financial statements in one coordinated call."""
        market_data = self.fetch_market_data(ticker, force_refresh=force_refresh)
        financials = self.fetch_financial_statements(ticker, force_refresh=force_refresh)
        
        all_warnings = list(market_data.warnings)
        if financials is None:
            all_warnings.append("Detailed financial statements could not be parsed; relying on basic quotes.")
        elif financials.free_cash_flow is None or financials.free_cash_flow <= 0:
            all_warnings.append("Free cash flow was not available or was negative in recent filings.")

        return DCFIngestionBundle(
            market_data=market_data,
            financials=financials,
            warnings=all_warnings
        )
