"""Feature engineering & stationarity enforcement for ARIMA.

ARIMA assumes (weak) stationarity of the modelled series. This module provides:

* ``log_transform`` – variance-stabilising log of a strictly-positive price series.
* ``log_returns``   – first difference of log prices (a common stationary target).
* ``adf_test``      – programmatic Augmented Dickey-Fuller test with a clean report.
* ``make_stationary`` – iteratively log-transform and difference until the ADF test
  rejects the unit-root null at the configured significance level (or a cap is hit).

The `d` implied here is informational: the model layer's ``auto_arima`` search
also estimates its own differencing order (via ADF). We surface the ADF-driven
order here so the analyst can sanity-check the two against each other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationarityReport:
    """Outcome of an Augmented Dickey-Fuller test."""

    statistic: float
    p_value: float
    used_lag: int
    n_obs: int
    critical_values: dict[str, float]
    is_stationary: bool
    significance: float

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "STATIONARY" if self.is_stationary else "NON-stationary"
        return (
            f"ADF: stat={self.statistic:.4f}, p={self.p_value:.4g} "
            f"(α={self.significance}) → {verdict}"
        )


@dataclass
class StationarityResult:
    """Result of forcing a series to stationarity."""

    series: pd.Series
    n_differences: int
    log_transformed: bool
    report: StationarityReport
    history: list[StationarityReport] = field(default_factory=list)


def log_transform(prices: pd.Series) -> pd.Series:
    """Return log(prices). Raises if any value is non-positive."""
    if (prices <= 0).any():
        raise ValueError(
            "log_transform requires strictly positive values; found "
            f"{int((prices <= 0).sum())} non-positive observation(s)."
        )
    return np.log(prices).rename(f"log_{prices.name}" if prices.name else "log_price")


def log_returns(prices: pd.Series) -> pd.Series:
    """Return log returns: diff of log prices. First observation is dropped."""
    return log_transform(prices).diff().dropna().rename(
        f"logret_{prices.name}" if prices.name else "log_return"
    )


def adf_test(
    series: pd.Series,
    *,
    significance: float | None = None,
    autolag: str = "AIC",
) -> StationarityReport:
    """Run the Augmented Dickey-Fuller test and return a structured report.

    Null hypothesis: the series has a unit root (is non-stationary). We reject it
    (i.e. declare stationarity) when ``p_value < significance``.
    """
    settings = get_settings()
    alpha = significance if significance is not None else settings.adf_significance

    clean = series.dropna()
    if len(clean) < 10:
        raise ValueError(f"ADF test needs ≥10 observations; got {len(clean)}.")

    stat, p_value, used_lag, n_obs, crit, _ = adfuller(clean, autolag=autolag)
    report = StationarityReport(
        statistic=float(stat),
        p_value=float(p_value),
        used_lag=int(used_lag),
        n_obs=int(n_obs),
        critical_values={k: float(v) for k, v in crit.items()},
        is_stationary=bool(p_value < alpha),
        significance=alpha,
    )
    logger.debug("%s", report)
    return report


def make_stationary(
    prices: pd.Series,
    *,
    settings: Settings | None = None,
    apply_log: bool = True,
) -> StationarityResult:
    """Force a price series to stationarity by log + iterative differencing.

    Procedure
    ---------
    1. Optionally log-transform (variance stabilisation), as specified in the thesis.
    2. Test with ADF. If p-value > α, difference once and re-test.
    3. Repeat until stationary or ``settings.max_differences`` is reached.

    Returns the transformed series plus a full audit trail of ADF reports.
    """
    settings = settings or get_settings()
    alpha = settings.adf_significance
    history: list[StationarityReport] = []

    working = log_transform(prices) if apply_log else prices.copy()
    if apply_log:
        logger.info("Applied log transform for variance stabilisation.")

    d = 0
    report = adf_test(working, significance=alpha)
    history.append(report)
    logger.info("d=%d | %s", d, report)

    while not report.is_stationary and d < settings.max_differences:
        working = working.diff().dropna()
        d += 1
        report = adf_test(working, significance=alpha)
        history.append(report)
        logger.info("d=%d | %s", d, report)

    if not report.is_stationary:
        logger.warning(
            "Series still non-stationary after d=%d (p=%.4g ≥ α=%.2f). "
            "Proceeding, but interpret results with caution.",
            d,
            report.p_value,
            alpha,
        )

    return StationarityResult(
        series=working,
        n_differences=d,
        log_transformed=apply_log,
        report=report,
        history=history,
    )
