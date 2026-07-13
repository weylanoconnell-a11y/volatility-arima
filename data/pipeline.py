"""Provider-agnostic historical-price ingestion.

The pipeline delegates the actual API/SDK call to a pluggable *provider*
(see ``data/providers.py``) and owns everything that is provider-independent: 

The default provider is ``yfinance``, which needs **no API key**, so the whole
pipeline runs out of the box. Switch providers via ``APP_DATA_PROVIDER`` (e.g.
``fmp`` or ``alpaca``) — see ``config/settings.py``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from config.settings import Settings, get_settings
from data.providers import MarketDataProvider, get_provider

logger = logging.getLogger(__name__)

# Columns we standardise on for every returned frame (ordered).
_SCHEMA = ["open", "high", "low", "close",
           "adj_close", "volume", "vwap", "trade_count"]


class DataPipeline:
    """Fetch, clean, cache and return historical bars for one or more symbols.

    Parameters
    ----------
    settings:
        Optional pre-built settings; defaults to the cached ``get_settings()``.
    provider:
        Optional provider override; defaults to the one named in ``settings``.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        provider: MarketDataProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.provider = provider or get_provider(self.settings)
        self.cache_dir = Path(self.settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------- cache
    def _cache_path(self, symbol: str) -> Path:
        end = self.settings.end_date.isoformat() if self.settings.end_date else "latest"
        raw = (
            f"{self.provider.name}|{symbol}|{self.settings.timeframe}"
            f"|{self.settings.start_date.isoformat()}|{end}|{self.settings.timezone}"
        )
        digest = hashlib.sha1(raw.encode()).hexdigest()[:12]
        return self.cache_dir / f"{self.provider.name}_{symbol}_{self.settings.timeframe}_{digest}.parquet"

    # ------------------------------------------------------------------- fetch
    def fetch_bars(self, symbol: str, *, force_refresh: bool = False) -> pd.DataFrame:
        """Return a cleaned, tz-aware OHLCV frame for ``symbol``.

        Parameters
        ----------
        symbol:
            Equity ticker (e.g. ``"CEG"``).
        force_refresh:
            Ignore any cache entry and re-hit the provider.
        """
        symbol = symbol.upper()
        cache_path = self._cache_path(symbol)

        if self.settings.use_cache and not force_refresh and cache_path.exists():
            logger.info("Loading %s bars from cache: %s",
                        symbol, cache_path.name)
            return pd.read_parquet(cache_path)

        logger.info(
            "Fetching %s %s bars via '%s' (%s → %s)",
            symbol,
            self.settings.timeframe,
            self.provider.name,
            self.settings.start_date,
            self.settings.end_date or "latest",
        )
        raw = self.provider.fetch(symbol)
        clean = self._transform(raw, symbol)

        if self.settings.use_cache:
            clean.to_parquet(cache_path)
            logger.debug("Cached %d rows to %s", len(clean), cache_path.name)
        return clean

    # --------------------------------------------------------------- transform
    def _transform(self, raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalise a raw provider frame: tz, schema, gaps, dedupe."""
        df = raw.copy()

        # Normalise the index to the configured market timezone, whether the
        # provider handed us tz-aware (Alpaca/intraday) or tz-naive (yfinance
        # daily, FMP daily) timestamps.
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            idx = idx.tz_localize(self.settings.timezone)
        else:
            idx = idx.tz_convert(self.settings.timezone)
        df.index = idx
        df.index.name = "timestamp"
        df = df.sort_index()

        # Keep a stable column set; some fields (vwap/trade_count/adj_close) vary.
        keep = [c for c in _SCHEMA if c in df.columns]
        if "close" not in keep:
            raise ValueError(
                f"Provider '{self.provider.name}' returned no 'close' column for "
                f"{symbol} (got {list(df.columns)})."
            )
        df = df[keep].astype("float64", errors="ignore")

        df = self._impute_gaps(df)
        df = df[~df.index.duplicated(keep="last")]

        logger.info(
            "%s: %d clean rows (%s → %s)",
            symbol,
            len(df),
            df.index.min().date(),
            df.index.max().date(),
        )
        return df

    def _impute_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill missing observations across the implied trading calendar.

        For daily bars we reindex onto US business days between the first and last
        observation, which surfaces holidays/halts as NaNs before we forward-fill.
        Volume is filled with 0 (no trading) rather than carried forward.
        """
        price_cols = [c for c in df.columns if c != "volume"]

        if self.settings.timeframe != "1Day":
            # For intraday frames, only ffill the sparse gaps already present.
            df[price_cols] = df[price_cols].ffill()
            if "volume" in df.columns:
                df["volume"] = df["volume"].fillna(0)
            return df.dropna(subset=["close"])

        full_index = pd.bdate_range(
            start=df.index.min(), end=df.index.max(), tz=self.settings.timezone
        )
        n_before = len(df)
        df = df.reindex(full_index)
        n_gaps = int(df["close"].isna().sum())

        df[price_cols] = df[price_cols].ffill()
        if "volume" in df.columns:
            df["volume"] = df["volume"].fillna(0)

        df = df.dropna(subset=["close"])  # drop any leading pre-history NaNs
        df.index.name = "timestamp"

        if n_gaps:
            logger.info(
                "Imputed %d calendar gaps via forward-fill (%d → %d rows)",
                n_gaps,
                n_before,
                len(df),
            )
        return df

    # ------------------------------------------------------------------- multi
    def fetch_universe(self, *, include_context: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch the primary ticker and (optionally) the context instruments."""
        symbols = [self.settings.ticker]
        if include_context:
            symbols += list(self.settings.context_tickers)

        out: dict[str, pd.DataFrame] = {}
        for sym in symbols:
            try:
                out[sym] = self.fetch_bars(sym)
            except Exception as exc:  # keep going if a context symbol fails
                if sym == self.settings.ticker:
                    raise
                logger.warning("Skipping context symbol %s: %s", sym, exc)
        return out


# Backward-compatible alias: earlier code/notebooks referenced ``AlpacaDataPipeline``.
# The pipeline is now provider-agnostic, but the name is kept so imports don't break.
AlpacaDataPipeline = DataPipeline


def load_prices(
    symbol: str | None = None,
    *,
    column: str = "close",
    settings: Settings | None = None,
) -> pd.Series:
    """Convenience: return a single, named price ``Series`` for modelling.

    Defaults to the configured primary ticker's close price.
    """
    settings = settings or get_settings()
    symbol = (symbol or settings.ticker).upper()
    pipeline = DataPipeline(settings)
    df = pipeline.fetch_bars(symbol)
    if column not in df.columns:
        raise KeyError(
            f"Column {column!r} not in fetched data ({list(df.columns)}).")
    return df[column].rename(f"{symbol}_{column}")


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    s = get_settings()
    prices = load_prices()
    print(prices.tail(10))
    print(
        f"\nLoaded {len(prices)} rows for {s.ticker} via '{s.data_provider}'.")
