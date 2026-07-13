"""ARIMA modelling: dynamic order selection + walk-forward backtesting.

Key decisions
-------------
* Orders (p, d, q). A stepwise, AIC-driven search
  (Hyndman-Khandakar style) selects the order over the space defined in
  ``Settings`` - implemented directly on ``statsmodels`` so the project runs on
  modern toolchains (numpy 2.x / Python 3.13), where ``pmdarima`` is unsupported.
* Seasonal components are disabled (``seasonal=False``) per the thesis spec - the
  search space is non-seasonal ARIMA(p, d, q) only.
* Evaluation uses **walk-forward validation** (expanding or rolling window) rather
  than a single train/test split, which would leak look-ahead information and give
  an over-optimistic view of out-of-sample skill.
* We model **log prices** and forecast them, then exponentiate back to price space
  for reporting, so metrics are in dollars and comparable to a naïve benchmark.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

Order = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """Container for walk-forward validation output."""

    predictions: pd.Series
    actuals: pd.Series
    naive_predictions: pd.Series
    orders: list[Order]
    horizon: int
    expanding: bool

    @property
    def errors(self) -> pd.Series:
        return (self.actuals - self.predictions).rename("error")

    @property
    def aligned(self) -> pd.DataFrame:
        """Actual / model / naïve, aligned on the shared index."""
        return pd.DataFrame(
            {
                "actual": self.actuals,
                "predicted": self.predictions,
                "naive": self.naive_predictions,
            }
        ).dropna()


# ---------------------------------------------------------------------------
# Low-level fitting helpers
# ---------------------------------------------------------------------------
def _fit_arima(y: np.ndarray, order: Order):
    """Fit a single ARIMA(order) on ``y``; return the results object or ``None``.

    Failures (non-convergence, singular matrices) are swallowed and reported as a
    ``None`` fit so the stepwise search can simply skip that candidate.
    """
    p, d, q = order
    trend = "c" if d == 0 else "n"  # a constant only makes sense at d == 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        warnings.simplefilter("ignore", ValueWarning)
        warnings.simplefilter("ignore", RuntimeWarning)
        # Benign: statsmodels falls back to zero starting params for non-stationary/
        # non-invertible seeds while the search explores candidate orders.
        warnings.simplefilter("ignore", UserWarning)
        try:
            model = ARIMA(y, order=order, trend=trend)
            return model.fit()
        except (ValueError, np.linalg.LinAlgError, IndexError):
            return None


def _criterion(res, which: str) -> float:
    """Return the requested information criterion (falls back to AIC for 'oob')."""
    which = "aic" if which == "oob" else which
    value = getattr(res, which, None)
    return float(value) if value is not None and np.isfinite(value) else float("inf")


def _estimate_d(y: np.ndarray, max_d: int, alpha: float) -> int:
    """Estimate the differencing order via successive ADF tests (capped at max_d)."""
    d = 0
    current = np.asarray(y, dtype=float)
    while d < max_d:
        if len(current) < 10 or np.allclose(current, current[0]):
            break
        try:
            p_value = adfuller(current, autolag="AIC")[1]
        except Exception:  # pragma: no cover - degenerate series
            break
        if p_value < alpha:
            break
        current = np.diff(current)
        d += 1
    return d


def auto_arima(
    y: np.ndarray,
    *,
    settings: Settings,
) -> tuple[Order, object]:
    """Stepwise, IC-minimising ARIMA order search (non-seasonal).

    Differencing ``d`` is chosen by ADF; then ``p`` and ``q`` are searched with a
    hill-climbing stepwise procedure that starts from a small seed set and walks to
    neighbouring orders while the information criterion keeps improving. Returns the
    chosen ``(p, d, q)`` and the fitted ``statsmodels`` results object.
    """
    ic = settings.information_criterion
    max_p, max_q = settings.max_p, settings.max_q
    d = _estimate_d(y, settings.max_d, settings.adf_significance)

    cache: dict[tuple[int, int], tuple[float, object]] = {}

    def evaluate(p: int, q: int) -> float:
        if not (0 <= p <= max_p and 0 <= q <= max_q):
            return float("inf")
        if (p, q) in cache:
            return cache[(p, q)][0]
        res = _fit_arima(y, (p, d, q))
        score = _criterion(res, ic) if res is not None else float("inf")
        cache[(p, q)] = (score, res)
        return score

    # Seed set (Hyndman-Khandakar-style), clipped to the configured bounds.
    seeds = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 2)]
    for p, q in seeds:
        evaluate(min(p, max_p), min(q, max_q))

    best_pq = min(cache, key=lambda k: cache[k][0])

    # Hill-climb over neighbours until no local improvement.
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1),
             (-1, -1), (1, 1), (-1, 1), (1, -1)]
    improved = True
    while improved:
        improved = False
        bp, bq = best_pq
        best_score = cache[best_pq][0]
        for dp, dq in steps:
            cand = (bp + dp, bq + dq)
            if evaluate(*cand) < best_score - 1e-6:
                best_pq = cand
                improved = True
        # loop re-reads best_pq; the cache prevents any re-fitting

    best_res = cache[best_pq][1]
    if best_res is None:  # everything failed → fall back to a random walk
        order: Order = (0, max(d, 1), 0)
        best_res = _fit_arima(y, order)
        logger.warning(
            "auto_arima: all candidates failed; falling back to %s", order)
        return order, best_res

    order = (best_pq[0], d, best_pq[1])
    logger.info("Selected ARIMA order %s (%s=%.2f)",
                order, ic, cache[best_pq][0])
    return order, best_res


# ---------------------------------------------------------------------------
# Forecaster wrapper
# ---------------------------------------------------------------------------
class ArimaForecaster:
    """Fits an auto-selected ARIMA on a (log-)price series and forecasts it.

    The chosen order is exposed via :pyattr:`order` after :meth:`fit`.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._res = None
        self.order: Order | None = None

    def fit(self, series: pd.Series) -> "ArimaForecaster":
        """Select the order via :func:`auto_arima` and fit (typically on log price)."""
        clean = series.dropna()
        logger.info(
            "auto_arima search on %d obs (max_p=%d, max_d=%d, max_q=%d, ic=%s)",
            len(clean),
            self.settings.max_p,
            self.settings.max_d,
            self.settings.max_q,
            self.settings.information_criterion,
        )
        self.order, self._res = auto_arima(
            clean.to_numpy(dtype=float), settings=self.settings)
        return self

    def forecast(self, steps: int) -> np.ndarray:
        """Return a ``steps``-length point forecast (same space as the fit series)."""
        if self._res is None:
            raise RuntimeError("Call fit() before forecast().")
        return np.asarray(self._res.forecast(steps=steps))

    def update(self, new_obs: float | np.ndarray) -> None:
        """Cheaply incorporate new observations, keeping the fitted parameters."""
        if self._res is None:
            raise RuntimeError("Call fit() before update().")
        self._res = self._res.append(np.atleast_1d(
            new_obs).astype(float), refit=False)

    def summary(self) -> str:
        if self._res is None:
            raise RuntimeError("Call fit() before summary().")
        return str(self._res.summary())


# ---------------------------------------------------------------------------
# Walk-forward backtest
# ---------------------------------------------------------------------------
def walk_forward_validate(
    prices: pd.Series,
    *,
    settings: Settings | None = None,
    model_in_log_space: bool = True,
) -> BacktestResult:
    """Run an expanding/rolling walk-forward backtest.

    At each step the model is trained on all data up to time ``t`` and asked to
    forecast ``t + horizon``. The order is re-selected via :func:`auto_arima` every
    ``settings.refit_every`` steps; in between, the fitted model is cheaply extended
    with the newly-arrived observation (``statsmodels`` ``append(refit=False)``),
    balancing realism against runtime.

    A **naïve random-walk benchmark** (forecast = last known price) is computed
    alongside so the ARIMA's marginal skill can be judged honestly - for daily
    equity prices this benchmark is notoriously hard to beat.
    """
    settings = settings or get_settings()
    horizon = settings.forecast_horizon
    window0 = settings.backtest_window
    expanding = settings.expanding_window

    prices = prices.dropna()
    n = len(prices)
    if n <= window0 + horizon:
        raise ValueError(
            f"Not enough data ({n} obs) for backtest_window={window0} + "
            f"forecast_horizon={horizon}. Fetch a longer history or lower the window."
        )

    work = np.log(prices.to_numpy(dtype=float)
                  ) if model_in_log_space else prices.to_numpy(dtype=float)
    index = prices.index

    preds: list[float] = []
    actuals: list[float] = []
    naive: list[float] = []
    pred_index: list[pd.Timestamp] = []
    orders: list[Order] = []

    res = None
    steps_since_refit = 0

    last_start = n - horizon
    for t in range(window0, last_start + 1):
        train = work[:t] if expanding else work[t - window0: t]

        need_refit = res is None or steps_since_refit >= settings.refit_every
        if need_refit:
            order, res = auto_arima(train, settings=settings)
            orders.append(order)
            steps_since_refit = 0
            logger.debug("t=%d refit → order=%s", t, order)
        else:
            # Extend with the single new observation revealed since the last step,
            # reusing the previously-estimated parameters (no re-search).
            res = res.append(work[t - 1: t], refit=False)
            steps_since_refit += 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ValueWarning)
            fc = float(np.asarray(res.forecast(steps=horizon))[-1])

        target_pos = t + horizon - 1
        preds.append(float(np.exp(fc)) if model_in_log_space else fc)
        actuals.append(float(prices.iloc[target_pos]))
        # random walk: carry last known price
        naive.append(float(prices.iloc[t - 1]))
        pred_index.append(index[target_pos])

    logger.info(
        "Walk-forward complete: %d forecasts, %d refits (%s window).",
        len(preds),
        len(orders),
        "expanding" if expanding else "rolling",
    )

    idx = pd.Index(pred_index, name="timestamp")
    return BacktestResult(
        predictions=pd.Series(preds, index=idx, name="predicted"),
        actuals=pd.Series(actuals, index=idx, name="actual"),
        naive_predictions=pd.Series(naive, index=idx, name="naive"),
        orders=orders,
        horizon=horizon,
        expanding=expanding,
    )
