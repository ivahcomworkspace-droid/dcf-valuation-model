"""
Unit tests for services.schemas and services.ingestion.
Validates multi-currency resolution, error boundaries, and TTL cache operations.
"""

from datetime import datetime, timezone
import pytest
from services.schemas import (
    MarketDataSnapshot,
    FinancialStatementItems,
    DCFIngestionBundle,
    get_currency_symbol,
)
from services.ingestion import MarketDataService, TickerNotFoundError


def test_currency_symbol_resolution():
    """Verifies that currency symbols are properly mapped for global markets."""
    assert get_currency_symbol("USD") == "$"
    assert get_currency_symbol("INR") == "₹"
    assert get_currency_symbol("EUR") == "€"
    assert get_currency_symbol("GBP") == "£"
    assert get_currency_symbol("JPY") == "¥"
    assert get_currency_symbol("CAD") == "CA$"
    assert get_currency_symbol("UNKNOWN") == "UNKNOWN "
    assert get_currency_symbol(None) == "$"


def test_market_data_snapshot_validation_with_currency():
    """Verifies snapshot creation and dynamic currency symbol property."""
    snapshot_us = MarketDataSnapshot(
        ticker="  aapl ",
        company_name="Apple Inc.",
        current_price=175.50,
        currency="USD",
        diluted_shares=15500.0,
        cash_and_equivalents=30000.0,
        total_debt=110000.0,
        beta=1.15
    )
    assert snapshot_us.ticker == "AAPL"
    assert snapshot_us.currency_symbol == "$"
    assert snapshot_us.net_debt == 80000.0

    snapshot_in = MarketDataSnapshot(
        ticker="tatasteel.ns",
        company_name="Tata Steel Limited",
        current_price=175.50,
        currency="INR",
        diluted_shares=12474.0,
        cash_and_equivalents=10000.0,
        total_debt=80000.0,
        beta=1.2
    )
    assert snapshot_in.ticker == "TATASTEEL.NS"
    assert snapshot_in.currency_symbol == "₹"


def test_market_data_service_caching():
    """Verifies in-memory TTL caching prevents redundant lookups."""
    service = MarketDataService(ttl_seconds=60)
    service.clear_cache()

    mock_snapshot = MarketDataSnapshot(
        ticker="MSFT",
        company_name="Microsoft Corporation",
        current_price=410.0,
        currency="USD",
        diluted_shares=7400.0,
        cash_and_equivalents=80000.0,
        total_debt=60000.0,
        beta=0.9
    )
    service._cache["MSFT"] = (datetime.now(timezone.utc), mock_snapshot, None)

    retrieved = service.fetch_market_data("msft")
    assert retrieved.ticker == "MSFT"
    assert retrieved.current_price == 410.0
    assert retrieved.currency_symbol == "$"


def test_empty_ticker_validation():
    """Verifies that an empty ticker raises a clean ValueError."""
    service = MarketDataService()
    with pytest.raises(ValueError, match="Ticker symbol cannot be empty"):
        service.fetch_market_data("   ")
