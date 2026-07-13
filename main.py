"""End-to-end volatility (VIX) ARIMA research pipeline.

Thesis
------
Equity *price levels* are efficient random walks - ARIMA cannot beat a naive
"no change" forecast on them.
The VIX index is strongly **mean-reverting**: volatility spikes on
shocks and decays back toward a long-run level. That mean reversion is exactly
the linear structure ARIMA's AR terms can exploit, so here the model is expected
to *beat* the random walk.

Because the VIX is a positive, mean-reverting *level* (not a price that trends),
we model it in **level space** (``log_space=False``) so ``auto_arima`` can pick
``d=0`` and estimate genuine autoregressive mean reversion.

Usage
-----
    python main.py                     # model ^VIX (default)
    python main.py --ticker "^VIX3M"   # 3-month VIX, or any Yahoo symbol
    python main.py --no-cache          # force a fresh pull

Steps: fetch -> stationarity check -> fit auto_arima -> walk-forward backtest ->
metrics + residual diagnostics -> save artefacts to outputs/.
"""

from __future__ import annotations
from src.model import ArimaForecaster, walk_forward_validate
from src.features import make_stationary
from src.evaluation import compute_metrics, ljung_box_test, plot_diagnostics
from data.pipeline import DataPipeline
from config.settings import configure_logging, get_settings

import argparse
import json
import logging
import re
import sys
from pathlib import Path

# Windows terminals default to cp1252, which cannot encode characters like "->".
for _stream in (sys.stdout, sys.stderr):
    try:
        # type: ignore[union-attr]
        _stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    except (AttributeError, ValueError):  # pragma: no cover
        pass


logger = logging.getLogger("volatility_arima")


def _safe(name: str) -> str:
    """Filename-safe version of a ticker (^VIX -> VIX)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("_") or "series"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the VIX mean-reversion ARIMA pipeline.")
    p.add_argument("--ticker", type=str, default=None,
                   help="Override the series symbol.")
    p.add_argument("--provider", type=str, default=None,
                   choices=["yfinance", "fmp", "alpaca"], help="Override the data provider.")
    p.add_argument("--log-space", dest="log_space", action="store_true",
                   help="Model log(series) instead of the level.")
    p.add_argument("--no-cache", action="store_true",
                   help="Bypass the on-disk cache.")
    p.add_argument("--no-show", action="store_true",
                   help="Do not open plot windows.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if args.ticker:
        settings.ticker = args.ticker.upper()
    if args.provider:
        settings.data_provider = args.provider
    if args.log_space:
        settings.log_space = True
    if args.no_cache:
        settings.use_cache = False
    configure_logging(settings.log_level)

    use_log = settings.log_space
    space = "log" if use_log else "level"
    out_dir = Path(settings.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe(settings.ticker)

    # 1) Ingest -------------------------------------------------------------
    logger.info("=== 1/5 Fetching %s via '%s' ===",
                settings.ticker, settings.data_provider)
    pipeline = DataPipeline(settings)
    df = pipeline.fetch_bars(settings.ticker)
    series = df["close"].rename(settings.ticker)

    # 2) Stationarity -------------------------------------------------------
    logger.info("=== 2/5 Stationarity analysis (%s space) ===", space)
    stat = make_stationary(series, settings=settings, apply_log=use_log)
    logger.info("Stationary after %d difference(s). Final %s",
                stat.n_differences, stat.report)

    # 3) Fit auto_arima on full history (for the reported order & summary) ---
    logger.info("=== 3/5 Fitting auto_arima on the %s series ===", space)
    import numpy as np

    fit_series = np.log(series) if use_log else series
    forecaster = ArimaForecaster(settings).fit(fit_series)
    logger.info("Chosen order: %s", forecaster.order)

    # 4) Walk-forward backtest ---------------------------------------------
    logger.info("=== 4/5 Walk-forward backtest ===")
    bt = walk_forward_validate(
        series, settings=settings, model_in_log_space=use_log)

    # 5) Evaluate & report --------------------------------------------------
    logger.info("=== 5/5 Evaluation & diagnostics ===")
    metrics = compute_metrics(bt.actuals, bt.predictions, bt.naive_predictions)
    lb = ljung_box_test(bt.errors)

    fig_path = out_dir / f"{tag}_diagnostics.png"
    plot_diagnostics(bt.actuals, bt.predictions, residuals=bt.errors,
                     save_path=fig_path, show=not args.no_show)

    report = {
        "ticker": settings.ticker,
        "model_space": space,
        "n_obs": int(len(series)),
        "stationarity": {
            "log_transformed": stat.log_transformed,
            "n_differences": stat.n_differences,
            "final_adf_p_value": stat.report.p_value,
            "is_stationary": stat.report.is_stationary,
        },
        "arima_order_full_sample": list(forecaster.order) if forecaster.order else None,
        "backtest": {
            "horizon": bt.horizon,
            "expanding_window": bt.expanding,
            "n_forecasts": int(len(bt.predictions)),
            "n_refits": len(bt.orders),
            "distinct_orders": sorted({tuple(o) for o in bt.orders}),
        },
        "metrics": metrics.as_dict(),
        "ljung_box_min_pvalue": float(lb["lb_pvalue"].min()),
    }
    report_path = out_dir / f"{tag}_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    logger.info("Wrote summary report to %s", report_path)

    print("\n" + "=" * 70)
    print(f"RESULT for {settings.ticker}  (modelled in {space} space)")
    print("=" * 70)
    print(metrics)
    print(f"Full-sample ARIMA order: {forecaster.order}")
    print(f"Artefacts: {report_path}  |  {fig_path}")
    if metrics.beats_naive:
        print(
            f"\nRESULT: ARIMA BEAT the naive random walk (skill={metrics.skill_vs_naive:.4f}). "
            "The VIX's mean reversion is real, exploitable autocorrelation — exactly "
            "the structure that is absent from equity price levels."
        )
    else:
        print(
            f"\nNOTE: ARIMA did not beat the random walk here (skill={metrics.skill_vs_naive:.4f}). "
            "Try modelling in level space (default) and a longer history."
        )


if __name__ == "__main__":
    main()
