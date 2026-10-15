"""
Signal logic shared by the backtest and the live runner.

SPY + dip-buy boost:

  Hold 100% SPY. When the RSI(2) dip rule fires on SPY, rotate into SSO
  (2x S&P 500) at the close; rotate back to SPY at the close once the dip
  rule exits.

  Dip rule (Connors & Alvarez, "Short Term Trading Strategies That Work",
  2008), computed on SPY's dividend-adjusted closes:
      enter when  close > SMA(200)  and  RSI(2) < 10
      exit  when  close > SMA(5)

The parameters are the published constants, not fitted to data. Don't tune
them against the backtest. live.py calls exactly the same target_weights()
that the backtest grades.
"""

import pandas as pd

CORE_SYMBOL = "SPY"
BOOST_SYMBOL = "SSO"

RSI_PERIOD = 2
RSI_ENTRY = 10.0
REGIME_SMA_DAYS = 200
EXIT_SMA_DAYS = 5


def universe() -> list[str]:
    return [CORE_SYMBOL, BOOST_SYMBOL]


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    # No losses in the window → inf → 100 already; 0/0 on a flat price → neutral
    return out.fillna(50.0).where(avg_gain.notna())


def dip_position(close: pd.Series) -> pd.Series:
    """True on each date the dip rule is long after that day's close."""
    regime = close.rolling(REGIME_SMA_DAYS).mean()
    exit_sma = close.rolling(EXIT_SMA_DAYS).mean()
    r = rsi(close)

    entry = ((close > regime) & (r < RSI_ENTRY)).to_numpy()
    exit_ = (close > exit_sma).to_numpy()

    held = []
    in_pos = False
    for e, x in zip(entry, exit_):
        if in_pos and x:
            in_pos = False
        elif not in_pos and e:
            in_pos = True
        held.append(in_pos)
    return pd.Series(held, index=close.index)


def target_weights(spy_close: pd.Series) -> pd.DataFrame:
    """Per-symbol portfolio weight to hold after each date's close."""
    boost = dip_position(spy_close).astype(float)
    return pd.DataFrame({CORE_SYMBOL: 1.0 - boost, BOOST_SYMBOL: boost}, index=spy_close.index)
