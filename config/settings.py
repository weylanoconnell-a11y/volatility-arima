"""Centralised, type-safe configuration.

All runtime configuration is defined here as a single ``Settings`` object backed
by ``pydantic-settings``. Values are resolved in this order of precedence:

1. Environment variables (e.g. ``FMP_API_KEY``, ``APP_TICKER``).
2. A local ``.env`` file (see ``.env.example``).
3. The defaults declared on the model below.

Import ``get_settings()`` anywhere in the codebase; it is cached so the ``.env``
file and environment are parsed exactly once per process.

Data providers
--------------
The project is provider-agnostic. ``data_provider`` selects the backend:

* ``yfinance`` (default) — Yahoo Finance, **no API key required**. Runs out of the box.
* ``fmp``      — Financial Modeling Prep, needs a (free) ``FMP_API_KEY``.
* ``alpaca``   — Alpaca Market Data v2, needs ``ALPACA_API_KEY`` / ``ALPACA_SECRET_KEY``.

"""

from __future__ import annotations

import logging
from datetime import date
from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# Timeframe strings we support; each provider maps these to its own vocabulary.
TimeframeStr = Literal["1Min", "5Min", "15Min", "1Hour", "1Day", "1Week"]
ProviderStr = Literal["yfinance", "fmp", "alpaca"]


class Settings(BaseSettings):
    """Strongly-typed application settings.

    The model keeps *secrets* (provider credentials, read from bare names like
    ``FMP_API_KEY``) separate from *tunables* (prefixed ``APP_``) so the
    hyper-parameters can be overridden freely without touching credentials.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --------------------------------------------------------------- provider
    data_provider: ProviderStr = Field(
        default="yfinance",
        validation_alias="APP_DATA_PROVIDER",
        description="Market-data backend. Default 'yfinance' needs no API key.",
    )

    # ------------------------------------------------------- credentials (opt.)
    # Financial Modeling Prep
    fmp_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="FMP_API_KEY",
        description="Financial Modeling Prep API key (needed only if provider='fmp').",
    )
    fmp_base_url: str = Field(
        default="https://financialmodelingprep.com/api/v3",
        validation_alias="FMP_BASE_URL",
        description="FMP API base URL.",
    )
    # Alpaca (kept for compatibility; optional)
    alpaca_api_key: SecretStr | None = Field(
        default=None,
        validation_alias="ALPACA_API_KEY",
        description="Alpaca API key id (needed only if provider='alpaca').",
    )
    alpaca_secret_key: SecretStr | None = Field(
        default=None,
        validation_alias="ALPACA_SECRET_KEY",
        description="Alpaca API secret key (needed only if provider='alpaca').",
    )
    alpaca_base_url: str = Field(
        default="https://data.alpaca.markets",
        validation_alias="ALPACA_BASE_URL",
        description="Alpaca market-data API base URL.",
    )

    # ---------------------------------------------------------------- universe
    ticker: str = Field(
        default="^VIX",
        validation_alias="APP_TICKER",
        description="Series to model (default: ^VIX, the CBOE volatility index).",
    )
    # Tradable volatility vehicles that track (imperfectly) the VIX complex.
    context_tickers: tuple[str, ...] = Field(
        default=("VIXY", "VXX", "UVXY"),
        validation_alias="APP_CONTEXT_TICKERS",
        description=(
            "Tradable VIX-linked ETFs/ETNs for context: VIXY, VXX, UVXY. Note the "
            "VIX index itself is not directly tradable — see README."
        ),
    )

    # ----------------------------------------------------------------- windows
    start_date: date = Field(
        default=date(2019, 1, 1),
        validation_alias="APP_START_DATE",
        description="Inclusive start date (VIX has deep history; widen for more data).",
    )
    end_date: date | None = Field(
        default=None,
        validation_alias="APP_END_DATE",
        description="Inclusive end date; None means 'up to the latest bar'.",
    )
    timeframe: TimeframeStr = Field(
        default="1Day",
        validation_alias="APP_TIMEFRAME",
        description="Bar aggregation granularity.",
    )
    timezone: str = Field(
        default="America/New_York",
        validation_alias="APP_TIMEZONE",
        description="IANA tz used to localise the index (US equity market tz).",
    )

    # ------------------------------------------------------------ model tuning
    # Search bounds for the auto_arima order search. Kept as config, NOT hardcoded.
    # Bounded modestly here so the daily walk-forward backtest stays tractable;
    # AR(1-3) already captures the bulk of VIX mean reversion.
    max_p: int = Field(default=3, validation_alias="APP_MAX_P", ge=0)
    max_q: int = Field(default=2, validation_alias="APP_MAX_Q", ge=0)
    max_d: int = Field(default=1, validation_alias="APP_MAX_D", ge=0)
    information_criterion: Literal["aic", "bic", "hqic", "oob"] = Field(
        default="aic", validation_alias="APP_IC"
    )
    seasonal: bool = Field(
        default=False,
        validation_alias="APP_SEASONAL",
        description="Disabled per thesis spec — daily equity data has no clean seasonality.",
    )
    log_space: bool = Field(
        default=False,
        validation_alias="APP_LOG_SPACE",
        description=(
            "If True, model log(series) (right for prices). For a mean-reverting, "
            "already-positive level like the VIX, modelling the LEVEL (False) lets "
            "auto_arima pick d=0 and capture mean reversion directly."
        ),
    )

    # ---------------------------------------------------------- backtest / eval
    # VIX is persistent day-to-day but mean-reverts over ~1 month; empirically the
    # ARIMA model only beats the random walk at ~21-day (monthly) horizons.
    forecast_horizon: int = Field(
        default=21,
        validation_alias="APP_FORECAST_HORIZON",
        ge=1,
        description="Steps ahead to forecast; ~21 trading days (monthly) for VIX.",
    )
    backtest_window: int = Field(
        default=252,
        validation_alias="APP_BACKTEST_WINDOW",
        ge=30,
        description="Initial training window size (≈ 1 trading year for daily bars).",
    )
    expanding_window: bool = Field(
        default=True,
        validation_alias="APP_EXPANDING_WINDOW",
        description="True = expanding window; False = rolling fixed-size window.",
    )
    refit_every: int = Field(
        default=21,
        validation_alias="APP_REFIT_EVERY",
        ge=1,
        description="Re-run auto_arima order selection every N backtest steps (~monthly).",
    )
    adf_significance: float = Field(
        default=0.05,
        validation_alias="APP_ADF_SIGNIFICANCE",
        gt=0.0,
        lt=1.0,
        description="p-value threshold below which a series is deemed stationary.",
    )
    max_differences: int = Field(
        default=2,
        validation_alias="APP_MAX_DIFFERENCES",
        ge=0,
        description="Cap on successive differencing when forcing stationarity.",
    )

    # ---------------------------------------------------------------- runtime
    cache_dir: str = Field(default="data/cache",
                           validation_alias="APP_CACHE_DIR")
    output_dir: str = Field(
        default="outputs", validation_alias="APP_OUTPUT_DIR")
    use_cache: bool = Field(default=True, validation_alias="APP_USE_CACHE")
    log_level: str = Field(default="INFO", validation_alias="APP_LOG_LEVEL")

    # --------------------------------------------------------------- validators
    @field_validator("ticker")
    @classmethod
    def _upper_ticker(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("data_provider", mode="before")
    @classmethod
    def _lower_provider(cls, v: str) -> str:
        return v.strip().lower() if isinstance(v, str) else v

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, v: str) -> str:
        v = v.upper()
        if v not in logging._nameToLevel:  # noqa: SLF001 - intentional lookup
            raise ValueError(f"Invalid log level: {v!r}")
        return v

    # ----------------------------------------------------------------- helpers
    def require_fmp_key(self) -> str:
        """Return the FMP key or raise a clear, actionable error."""
        if self.fmp_api_key is None:
            raise RuntimeError(
                "provider='fmp' but FMP_API_KEY is not set. Get a free key at "
                "https://site.financialmodelingprep.com/developer/docs and add "
                "FMP_API_KEY=... to your .env (or switch APP_DATA_PROVIDER=yfinance)."
            )
        return self.fmp_api_key.get_secret_value()

    def require_alpaca_credentials(self) -> tuple[str, str]:
        """Return (key, secret) for Alpaca or raise a clear, actionable error."""
        if self.alpaca_api_key is None or self.alpaca_secret_key is None:
            raise RuntimeError(
                "provider='alpaca' but ALPACA_API_KEY / ALPACA_SECRET_KEY are not set. "
                "Add them to your .env (or switch APP_DATA_PROVIDER=yfinance)."
            )
        return (
            self.alpaca_api_key.get_secret_value(),
            self.alpaca_secret_key.get_secret_value(),
        )


def configure_logging(level: str | int = "INFO") -> None:
    """Configure root logging once, with a consistent, readable format."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, validated ``Settings`` instance.

    Credentials are validated lazily by the selected provider, so this succeeds
    even with no keys set (the default ``yfinance`` provider needs none).
    """
    try:
        # type: ignore[call-arg]  # values sourced from env/.env
        settings = Settings()
    except Exception as exc:  # pragma: no cover - surfaced to the user directly
        raise RuntimeError(
            "Failed to load configuration. If you customised .env, check it against "
            f".env.example.\nUnderlying error: {exc}"
        ) from exc

    configure_logging(settings.log_level)
    logger.debug(
        "Loaded settings: provider=%s ticker=%s timeframe=%s",
        settings.data_provider,
        settings.ticker,
        settings.timeframe,
    )
    return settings
