# Volatility ARIMA - Forecasting the VIX (mean reversion)

## True Purpose

The model has potentially overfitted characteristics. The puporse isn't to utilze
ARIMA on the VIX scale, but rather use this model to properly trade VIX-matching
ETF's. Trading the VIX with the ARIMA data could create high autocorerection.

A research scaffold that models the **CBOE Volatility Index (VIX)** with ARIMA and
**beats a naïve random walk**

> **Headline result** (default config, `^VIX`, level space, 21-day horizon,
> 2019–present): ARIMA **skill_vs_naive = 0.979** — it beats the random walk by
> ~2%. The full-sample order is **(3, 0, 2)** - `d=0` with genuine AR/MA terms, i.e.
> real mean-reversion structure, not the `(0,1,0)` random walk that equity prices
> collapse to.

## Two Design Choices

1. **Level space, not log/price space** (`log_space=False`). Because the VIX is a
   positive, mean-reverting *level*, we model it directly so `auto_arima` selects
   `d=0` and estimates the mean reversion - instead of differencing it away.
2. **A ~1-month horizon.** The VIX is *persistent* day-to-day (a spike stays
   elevated for a while, so "no change" is a decent short-term guess), but mean
   reversion dominates over weeks. Empirically the model only beats the random
   walk at ~21 trading days:

   | Horizon | skill_vs_naive | Beats RandomWalk? |
   |---|---:|:--:|
   | 1 day | 1.02 | No |
   | 5 days | 1.07 | No |
   | 10 days | 1.05 | No |
   | **21 days** | **0.97** | YES |

   (Full sweep in `../etf_market_research/RESULTS.md`, Experiment 3.)

## Project structure

```
volatility_arima/
├── config/settings.py      # adds `log_space` + VIX-tuned defaults (^VIX, h=21)
├── data/                   # provider-agnostic ingestion (yfinance default, no key)
│   ├── providers.py
│   └── pipeline.py
├── src/
│   ├── features.py         # log/level transform, ADF, differencing
│   ├── model.py            # statsmodels stepwise auto_arima + walk-forward
│   └── evaluation.py       # RMSE/MAE/MAPE, skill_vs_naive, Ljung-Box, plots
├── main.py                 # end-to-end runner (honours `log_space`)
├── requirements.txt
└── outputs/                # VIX_report.json + VIX_diagnostics.png
```

The architecture is shared with the sibling projects; the volatility-specific
logic is the `log_space` switch (level modelling) and the VIX-tuned defaults.

## Configuration

Defaults live in [`config/settings.py`](config/settings.py); override via `.env`
(`APP_` prefix). Volatility-specific ones:

| Setting | Env var | Default | Meaning |
|---|---|---|---|
| Series | `APP_TICKER` | `^VIX` | The volatility index to model |
| Model space | `APP_LOG_SPACE` | `false` | `false` = level (mean reversion); `true` = log |
| Horizon | `APP_FORECAST_HORIZON` | `21` | ~1 month, where VIX reversion beats RW |
| Start | `APP_START_DATE` | `2019-01-01` | VIX has data back to 1990; widen for more |
| Search bounds | `APP_MAX_P/D/Q` | `3/1/2` | Kept modest so the daily backtest is tractable |

## Caveats

- **The VIX index is not directly tradable.** VIX-linked ETFs (VIXY, VXX, UVXY)
  suffer futures roll-decay and do **not** inherit this skill on their *price* - in
  the ETF screen VIXY was the *worst* performer. This project forecasts the index
  as a research signal; converting it to a P&L requires options or a futures model.
- **Skill is modest (~2%) and horizon-specific.** It's a real, statistically
  meaningful edge over the random walk, but small - honest for a single-series
  linear model. Ljung-Box still flags residual autocorrelation (partly the
  overlapping-horizon artifact discussed in the CEG project's README).
- Natural extensions: a **GARCH** volatility model, or ARIMAX using the VIX term
  structure (`^VIX` vs `^VIX3M`) as an exogenous mean-reversion signal.
