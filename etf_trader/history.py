"""
Historical daily bars for the backtest, from Yahoo Finance.

Yahoo goes back to each ETF's inception (Alpaca's free history starts in
2016, too short to cover 2008 or the 2011/2015 drawdowns). Prices are
dividend- and split-adjusted so returns are total returns. Cached to CSV so
reruns are reproducible and offline.
"""

from pathlib import Path

import pandas as pd

from . import strategy

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"


def load(refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (opens, closes), indexed by date, one column per symbol."""
    opens_path = CACHE_DIR / "opens.csv"
    closes_path = CACHE_DIR / "closes.csv"
    if not refresh and opens_path.exists() and closes_path.exists():
        read = lambda p: pd.read_csv(p, index_col=0, parse_dates=True)
        return read(opens_path), read(closes_path)

    import yfinance as yf

    raw = yf.download(
        strategy.universe(), start="1993-01-01", auto_adjust=True,
        progress=False, threads=False,
    )
    opens, closes = raw["Open"], raw["Close"]
    opens.index = opens.index.tz_localize(None)
    closes.index = closes.index.tz_localize(None)

    CACHE_DIR.mkdir(exist_ok=True)
    opens.to_csv(opens_path)
    closes.to_csv(closes_path)
    return opens, closes
