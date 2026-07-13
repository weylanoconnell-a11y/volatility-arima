"""Provider-agnostic data ingestion package (yfinance / FMP / Alpaca)."""

from data.pipeline import AlpacaDataPipeline, DataPipeline, load_prices
from data.providers import MarketDataProvider, get_provider

__all__ = [
    "DataPipeline",
    "AlpacaDataPipeline",  # backward-compatible alias
    "load_prices",
    "MarketDataProvider",
    "get_provider",
]
