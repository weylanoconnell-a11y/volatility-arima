"""Pluggable market-data providers.

The pipeline (``data/pipeline.py``) handles everything that is provider-agnostic:
timezone localisation, gap imputation, caching and column selection. This keeps
providers small and easy to add.

Providers
---------
* :class:`YFinanceProvider` — Yahoo Finance via ``yfinance``. **No API key.** Default.
* :class:`FMPProvider`      — Financial Modeling Prep REST API. Needs ``FMP_API_KEY``.
* :class:`AlpacaProvider`   — Alpaca Market Data v2 via ``alpaca-py``. Needs keys.

Heavy / optional third-party imports are done lazily inside each provider so that
importing this module never requires every SDK to be installed.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, timezone

import pandas as pd

from config.settings import Settings

logger = logging.getLogger(__name__)

# Canonical, ordered column set. Providers emit a subset; the pipeline trims.
CANONICAL_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "vwap",
    "trade_count",
]


class MarketDataProvider(ABC):
    """Abstract base for all data providers."""

    name: str = "base"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def fetch(self, symbol: str) -> pd.DataFrame:
        """Return raw bars for ``symbol`` between the configured start/end dates."""

    # -- shared helpers ------------------------------------------------------
    @property
    def _start(self) -> date:
        return self.settings.start_date

    @property
    def _end(self) -> date:
        return self.settings.end_date or datetime.now(tz=timezone.utc).date()

    @staticmethod
    def _empty_error(provider: str, symbol: str) -> ValueError:
        return ValueError(
            f"{provider} returned no data for {symbol!r}. Check the symbol, the date "
            f"range, and (for keyed providers) that your plan covers this history."
        )


# ---------------------------------------------------------------------------
# yfinance (default, keyless)
# ---------------------------------------------------------------------------
class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance provider — requires no API key."""

    name = "yfinance"

    _INTERVALS = {
        "1Min": "1m",
        "5Min": "5m",
        "15Min": "15m",
        "1Hour": "1h",
        "1Day": "1d",
        "1Week": "1wk",
    }

    def fetch(self, symbol: str) -> pd.DataFrame:
        import yfinance as yf

        interval = self._INTERVALS[self.settings.timeframe]
        # yfinance treats `end` as exclusive; add a day so the last date is included.
        end_exclusive = self._end + pd.Timedelta(days=1)

        logger.info(
            "yfinance: downloading %s [%s] %s → %s",
            symbol,
            interval,
            self._start,
            self._end,
        )
        raw = yf.download(
            tickers=symbol,
            start=self._start.isoformat(),
            end=end_exclusive.isoformat(),
            interval=interval,
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            raise self._empty_error("yfinance", symbol)

        df = raw.copy()
        # Single-symbol downloads can still return MultiIndex columns; flatten them.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )
        df.index = pd.to_datetime(df.index)
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        return df[keep].sort_index()


# ---------------------------------------------------------------------------
# Financial Modeling Prep (FMP)
# ---------------------------------------------------------------------------
class FMPProvider(MarketDataProvider):
    """Financial Modeling Prep provider (REST). Requires ``FMP_API_KEY``."""

    name = "fmp"

    # FMP intraday "historical-chart" interval slugs.
    _INTRADAY = {
        "1Min": "1min",
        "5Min": "5min",
        "15Min": "15min",
        "1Hour": "1hour",
    }

    def fetch(self, symbol: str) -> pd.DataFrame:
        import requests

        api_key = self.settings.require_fmp_key()
        base = self.settings.fmp_base_url.rstrip("/")
        tf = self.settings.timeframe

        if tf == "1Day":
            url = f"{base}/historical-price-full/{symbol}"
            params = {
                "from": self._start.isoformat(),
                "to": self._end.isoformat(),
                "apikey": api_key,
            }
            key = "historical"
        elif tf == "1Week":
            raise ValueError(
                "FMP path here supports '1Day' and intraday timeframes. For weekly "
                "bars, use APP_TIMEFRAME=1Day and resample, or APP_DATA_PROVIDER=yfinance."
            )
        else:
            slug = self._INTRADAY[tf]
            url = f"{base}/historical-chart/{slug}/{symbol}"
            params = {
                "from": self._start.isoformat(),
                "to": self._end.isoformat(),
                "apikey": api_key,
            }
            key = None  # intraday endpoint returns a bare list

        logger.info("fmp: GET %s (%s → %s)", url.split(
            "?")[0], self._start, self._end)
        resp = requests.get(url, params=params, timeout=30)
        self._raise_for_api_error(resp, symbol)

        payload = resp.json()
        records = payload.get(key, []) if key else payload
        if not records:
            raise self._empty_error("fmp", symbol)

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df = df.rename(columns={"adjClose": "adj_close"})
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        return df[keep]

    @staticmethod
    def _raise_for_api_error(resp, symbol: str) -> None:
        if resp.status_code == 401:
            raise RuntimeError(
                "FMP rejected the API key (HTTP 401). Check FMP_API_KEY.")
        if resp.status_code == 403:
            raise RuntimeError(
                "FMP returned HTTP 403 — your plan may not cover this endpoint/symbol."
            )
        resp.raise_for_status()
        # FMP signals some errors with a 200 + an error object.
        if isinstance(resp_json := _safe_json(resp), dict) and "Error Message" in resp_json:
            raise RuntimeError(
                f"FMP error for {symbol}: {resp_json['Error Message']}")


def _safe_json(resp):
    try:
        return resp.json()
    except Exception:  # pragma: no cover - non-JSON body
        return None


# ---------------------------------------------------------------------------
# Alpaca (optional)
# ---------------------------------------------------------------------------
class AlpacaProvider(MarketDataProvider):
    """Alpaca Market Data v2 provider via the modern ``alpaca-py`` SDK."""

    name = "alpaca"

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._client = None

    @property
    def _alpaca_client(self):
        if self._client is None:
            from alpaca.data.historical import StockHistoricalDataClient

            key, secret = self.settings.require_alpaca_credentials()
            self._client = StockHistoricalDataClient(
                api_key=key, secret_key=secret)
        return self._client

    def _timeframe(self):
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        mapping = {
            "1Min": TimeFrame(1, TimeFrameUnit.Minute),
            "5Min": TimeFrame(5, TimeFrameUnit.Minute),
            "15Min": TimeFrame(15, TimeFrameUnit.Minute),
            "1Hour": TimeFrame(1, TimeFrameUnit.Hour),
            "1Day": TimeFrame(1, TimeFrameUnit.Day),
            "1Week": TimeFrame(1, TimeFrameUnit.Week),
        }
        return mapping[self.settings.timeframe]

    def fetch(self, symbol: str) -> pd.DataFrame:
        from alpaca.data.requests import StockBarsRequest

        start = datetime.combine(
            self._start, datetime.min.time(), tzinfo=timezone.utc)
        end = (
            datetime.combine(self.settings.end_date,
                             datetime.max.time(), tzinfo=timezone.utc)
            if self.settings.end_date
            else None
        )
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=self._timeframe(),
            start=start,
            end=end,
        )
        bars = self._alpaca_client.get_stock_bars(request)
        df = bars.df
        if df is None or df.empty:
            raise self._empty_error("alpaca", symbol)
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level="symbol")
        df.index = pd.to_datetime(df.index, utc=True)
        keep = [c for c in CANONICAL_COLUMNS if c in df.columns]
        return df[keep].sort_index()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_PROVIDERS: dict[str, type[MarketDataProvider]] = {
    "yfinance": YFinanceProvider,
    "fmp": FMPProvider,
    "alpaca": AlpacaProvider,
}


def get_provider(settings: Settings) -> MarketDataProvider:
    """Instantiate the provider named by ``settings.data_provider``."""
    try:
        cls = _PROVIDERS[settings.data_provider]
    except KeyError as exc:  # pragma: no cover - guarded by settings Literal
        raise ValueError(
            f"Unknown data_provider {settings.data_provider!r}. "
            f"Choose one of: {', '.join(_PROVIDERS)}."
        ) from exc
    logger.debug("Using data provider: %s", cls.name)
    return cls(settings)
