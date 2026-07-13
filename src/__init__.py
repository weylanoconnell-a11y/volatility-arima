"""Core research package: features, modelling and evaluation."""

from src.features import (
    StationarityReport,
    adf_test,
    log_transform,
    make_stationary,
)
from src.model import ArimaForecaster, BacktestResult, walk_forward_validate
from src.evaluation import (
    ForecastMetrics,
    compute_metrics,
    ljung_box_test,
    plot_diagnostics,
)

__all__ = [
    "StationarityReport",
    "adf_test",
    "log_transform",
    "make_stationary",
    "ArimaForecaster",
    "BacktestResult",
    "walk_forward_validate",
    "ForecastMetrics",
    "compute_metrics",
    "ljung_box_test",
    "plot_diagnostics",
]
