"""Forecast evaluation: error metrics + residual diagnostics.

Metrics
-------
* RMSE, MAE, MAPE for the model and the naïve random-walk benchmark.
* A ``skill_vs_naive`` ratio: RMSE(model) / RMSE(naive). Values < 1 mean the model
  beats the benchmark; ≥ 1 means it does not.

Diagnostics
-----------
* Ljung-Box test for residual autocorrelation (well-specified residuals should be
  white noise → high p-values).
* Matplotlib figure with residuals, residual ACF, and predicted-vs-actual.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastMetrics:
    """Point-forecast accuracy metrics for model vs. naïve benchmark."""

    n: int
    rmse: float
    mae: float
    mape: float
    naive_rmse: float
    naive_mae: float
    naive_mape: float

    @property
    def skill_vs_naive(self) -> float:
        """RMSE(model)/RMSE(naive); < 1.0 means the model beats a random walk."""
        return float("inf") if self.naive_rmse == 0 else self.rmse / self.naive_rmse

    @property
    def beats_naive(self) -> bool:
        return self.skill_vs_naive < 1.0

    def as_dict(self) -> dict[str, float]:
        d = asdict(self)
        d["skill_vs_naive"] = self.skill_vs_naive
        d["beats_naive"] = self.beats_naive
        return d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        verdict = "BEATS" if self.beats_naive else "does NOT beat"
        return (
            f"n={self.n} | RMSE={self.rmse:.4f} MAE={self.mae:.4f} MAPE={self.mape:.2f}% "
            f"| naive RMSE={self.naive_rmse:.4f} | skill={self.skill_vs_naive:.3f} "
            f"→ model {verdict} random walk"
        )


def _rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def _mae(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)))


def _mape(actual: np.ndarray, pred: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def compute_metrics(
    actuals: pd.Series,
    predictions: pd.Series,
    naive_predictions: pd.Series | None = None,
) -> ForecastMetrics:
    """Compute RMSE/MAE/MAPE for the model and (optionally) a naïve benchmark."""
    df = pd.DataFrame({"actual": actuals, "pred": predictions})
    if naive_predictions is not None:
        df["naive"] = naive_predictions
    df = df.dropna()
    if df.empty:
        raise ValueError("No overlapping, non-NaN observations to score.")

    a = df["actual"].to_numpy()
    p = df["pred"].to_numpy()
    nv = df["naive"].to_numpy() if "naive" in df else a  # degenerate fallback

    metrics = ForecastMetrics(
        n=len(df),
        rmse=_rmse(a, p),
        mae=_mae(a, p),
        mape=_mape(a, p),
        naive_rmse=_rmse(a, nv),
        naive_mae=_mae(a, nv),
        naive_mape=_mape(a, nv),
    )
    logger.info("%s", metrics)
    return metrics


def ljung_box_test(residuals: pd.Series, *, lags: int = 10) -> pd.DataFrame:
    """Ljung-Box test for autocorrelation in residuals.

    Null hypothesis: residuals are independently distributed (white noise). A high
    p-value (> 0.05) is *good* - it means we fail to reject whiteness.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox

    res = residuals.dropna()
    result = acorr_ljungbox(res, lags=lags, return_df=True)
    worst_p = float(result["lb_pvalue"].min())
    if worst_p < 0.05:
        logger.warning(
            "Ljung-Box: residual autocorrelation detected (min p=%.4g < 0.05). "
            "Model may be mis-specified.",
            worst_p,
        )
    else:
        logger.info(
            "Ljung-Box: residuals look like white noise (min p=%.4g).", worst_p)
    return result


def plot_diagnostics(
    actuals: pd.Series,
    predictions: pd.Series,
    *,
    residuals: pd.Series | None = None,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Produce a 2x2 diagnostic figure and optionally save it.

    Panels: predicted-vs-actual over time, scatter of predicted vs actual,
    residual series, and residual ACF.
    """
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf

    resid = residuals if residuals is not None else (
        actuals - predictions).dropna()

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("ARIMA Forecast Diagnostics", fontsize=14, fontweight="bold")

    # (0,0) time series overlay
    ax = axes[0, 0]
    actuals.plot(ax=ax, label="Actual", linewidth=1.2)
    predictions.plot(ax=ax, label="Predicted", linewidth=1.2, alpha=0.8)
    ax.set_title("Predicted vs Actual")
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) scatter with 45° line
    ax = axes[0, 1]
    ax.scatter(actuals, predictions, s=12, alpha=0.5)
    lo = float(min(actuals.min(), predictions.min()))
    hi = float(max(actuals.max(), predictions.max()))
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Calibration")
    ax.grid(alpha=0.3)

    # (1,0) residuals
    ax = axes[1, 0]
    resid.plot(ax=ax, linewidth=0.9)
    ax.axhline(0, color="red", linewidth=0.8)
    ax.set_title("Residuals")
    ax.grid(alpha=0.3)

    # (1,1) residual ACF
    ax = axes[1, 1]
    plot_acf(resid.dropna(), ax=ax, lags=min(20, len(resid) // 2 - 1))
    ax.set_title("Residual ACF")

    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        logger.info("Saved diagnostics figure to %s", save_path)
    if show:
        plt.show()
    return fig
