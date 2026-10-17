"""
Pydantic schemas for data validation across market data, financial statements, and valuation ingestion.
Includes multi-currency symbol mapping and metadata warnings for incomplete data feeds.
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict
from pydantic import BaseModel, Field, field_validator


CURRENCY_MAP: Dict[str, str] = {
    "USD": "$",
    "INR": "₹",
    "EUR": "€",
    "GBP": "£",
    "GBp": "£",
    "JPY": "¥",
    "CAD": "CA$",
    "AUD": "A$",
    "CHF": "CHF ",
    "CNY": "¥",
    "HKD": "HK$",
    "SGD": "S$",
    "SEK": "kr ",
    "NOK": "kr ",
    "BRL": "R$",
    "KRW": "₩",
    "TWD": "NT$",
    "ZAR": "R ",
}


def get_currency_symbol(currency_code: Optional[str]) -> str:
    """Returns the recognized currency symbol or the formatted currency code."""
    if not currency_code:
        return "$"
    clean_code = currency_code.strip().upper()
    return CURRENCY_MAP.get(clean_code, f"{clean_code} ")


class MarketDataSnapshot(BaseModel):
    """Validated market data snapshot fetched from live feeds."""
    ticker: str = Field(..., description="Stock ticker symbol (e.g. AAPL, TATASTEEL.NS)")
    company_name: Optional[str] = Field(default=None, description="Legal company name")
    current_price: float = Field(..., gt=0, description="Latest share price in reporting currency")
    currency: str = Field(default="USD", description="Reporting currency (e.g. USD, INR, EUR)")
    diluted_shares: float = Field(..., gt=0, description="Diluted shares outstanding (Millions)")
    market_cap: Optional[float] = Field(default=None, description="Market Capitalization (Millions)")
    cash_and_equivalents: float = Field(default=0.0, ge=0, description="Cash & Short-Term Investments (Millions)")
    total_debt: float = Field(default=0.0, ge=0, description="Total short & long-term debt (Millions)")
    beta: Optional[float] = Field(default=1.0, description="5-Year Monthly Beta against broad market")
    interest_expense: Optional[float] = Field(default=None, description="Annual interest expense (Millions)")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of snapshot")
    warnings: List[str] = Field(default_factory=list, description="Data completeness notices or warnings")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @property
    def currency_symbol(self) -> str:
        """Dynamically resolved currency symbol (e.g. ₹ for INR, $ for USD)."""
        return get_currency_symbol(self.currency)

    @property
    def net_debt(self) -> float:
        """Net Debt = Total Debt - Cash & Equivalents (Millions)"""
        return self.total_debt - self.cash_and_equivalents


class FinancialStatementItems(BaseModel):
    """Key financial statement line items required for valuation and DuPont decomposition."""
    ticker: str = Field(..., description="Stock ticker symbol")
    fiscal_year: Optional[int] = Field(default=None, description="Fiscal year (e.g. 2024)")
    currency: str = Field(default="USD", description="Reporting currency")
    revenue: float = Field(..., gt=0, description="Total annual revenue (Millions)")
    net_income: float = Field(..., description="Net income (Millions)")
    total_assets: float = Field(..., gt=0, description="Total balance sheet assets (Millions)")
    total_equity: float = Field(..., description="Total shareholders' equity (Millions)")
    operating_cash_flow: Optional[float] = Field(default=None, description="Cash flow from operations (Millions)")
    capital_expenditures: Optional[float] = Field(default=None, description="Capital expenditures (Millions)")
    free_cash_flow: Optional[float] = Field(default=None, description="Free cash flow (Millions)")
    ebit: Optional[float] = Field(default=None, description="Earnings Before Interest & Taxes (Millions)")

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, v: str) -> str:
        return v.strip().upper()


class DCFIngestionBundle(BaseModel):
    """Consolidated bundle ready for transfer into fin_engine domain models."""
    market_data: MarketDataSnapshot
    financials: Optional[FinancialStatementItems] = None
    warnings: List[str] = Field(default_factory=list)

    def infer_base_fcff(self) -> float:
        """Determines base FCFF from financials if available, or estimates from OCF - CapEx."""
        if self.financials and self.financials.free_cash_flow is not None and self.financials.free_cash_flow > 0:
            return self.financials.free_cash_flow
        if self.financials and self.financials.operating_cash_flow is not None and self.financials.capital_expenditures is not None:
            fcf = self.financials.operating_cash_flow - abs(self.financials.capital_expenditures)
            if fcf > 0:
                return fcf
        if self.financials and self.financials.net_income is not None and self.financials.net_income > 0:
            return self.financials.net_income
        return 100.0
