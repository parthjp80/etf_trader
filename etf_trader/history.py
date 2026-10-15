"""
Historical daily bars for the backtest, from Yahoo Finance.

Yahoo goes back to each ETF's inception. Alpaca's free history starts in
2016, too short to cover 2008. Prices are dividend- and split-adjusted, so
returns are total returns. Results are cached to CSV so reruns are
reproducible and work offline.

SSO only exists from June 2006. For longer history, synthetic_2x() builds a
2x S&P series from SPY: each day it earns 2x SPY's return, minus financing
on the borrowed 1x and the expense ratio. The financing spread is set so
the simulation matches real SSO since 2006 (run_backtest prints the check).
"""

from pathlib import Path

import pandas as pd

from . import strategy

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
SYMBOLS = [strategy.CORE_SYMBOL, strategy.BOOST_SYMBOL, "^IRX"]

SSO_EXPENSE_RATIO = 0.0089
# Swap financing costs more than the T-bill rate. Without this spread the
# simulation overstates real SSO by ~0.8%/yr.
FINANCING_SPREAD = 0.008


def load(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (opens, closes), indexed by date, one column per symbol.

    The ^IRX column (13-week T-bill yield, %) appears in closes only.
    """
    opens_path = CACHE_DIR / "opens.csv"
    closes_path = CACHE_DIR / "closes.csv"
    if not refresh and opens_path.exists() and closes_path.exists():
        read = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)
        return read(opens_path), read(closes_path)

    import yfinance as yf

    raw = yf.download(SYMBOLS, start="1993-01-01", auto_adjust=True,
                      progress=False, threads=False)
    opens = raw["Open"].drop(columns="^IRX")
    closes = raw["Close"]
    for df in (opens, closes):
        df.index = df.index.tz_localize(None)
    opens, closes = opens.loc[closes["SPY"].notna()], closes.loc[closes["SPY"].notna()]

    CACHE_DIR.mkdir(exist_ok=True)
    opens.to_csv(opens_path)
    closes.to_csv(closes_path)
    return opens, closes


def synthetic_2x(spy: pd.Series, tbill_pct: pd.Series) -> pd.Series:
    rate = (tbill_pct.reindex(spy.index).ffill().fillna(0) / 100 + FINANCING_SPREAD) / 252
    daily = 2 * spy.pct_change() - rate - SSO_EXPENSE_RATIO / 252
    return (1 + daily.fillna(0)).cumprod() * 100
