"""
Signal logic shared by the backtest and the live runner.

Everything here is a pure function of daily (dividend-adjusted) closes, so
the backtest and the live bot cannot drift apart: live.py calls exactly the
same target_weights() the backtest is graded on.

Two sleeves, each 50% of equity:

  Mean reversion (SPY, QQQ — 25% each)
      Enter at the close when  close > SMA(200)  and  RSI(2) < 10.
      Exit at the close when   close > SMA(5).
      Rules from Connors & Alvarez, "Short Term Trading Strategies That
      Work" (2008). No stop loss — the SMA(5) exit is the only exit.

  Trend (SPY, EFA, TLT, GLD, cash = SHY)
      On the last trading day of each month, pick the asset with the best
      trailing 252-day return. Hold it if its close is above its 10-month
      SMA of month-end closes; otherwise hold SHY. Hold until next month-end.
      Momentum ranking per Antonacci (2014), SMA filter per Faber (2007).

Parameters are fixed constants taken from the published rules, not fitted
to data. Do not tune them against the backtest.
"""

import pandas as pd

MR_SYMBOLS = ("SPY", "QQQ")
TREND_SYMBOLS = ("SPY", "EFA", "TLT", "GLD")
CASH_SYMBOL = "SHY"

MR_WEIGHT = 0.5
TREND_WEIGHT = 0.5

RSI_PERIOD = 2
RSI_ENTRY = 10.0
REGIME_SMA_DAYS = 200
EXIT_SMA_DAYS = 5

MOMENTUM_DAYS = 252
TREND_SMA_MONTHS = 10


def universe() -> list[str]:
    return sorted(set(MR_SYMBOLS) | set(TREND_SYMBOLS) | {CASH_SYMBOL})


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    # No losses in the window → RSI 100 (division gives inf → 100 already;
    # 0/0 when price is flat gives NaN, treat as neutral)
    return out.fillna(50.0).where(avg_gain.notna())


def mean_reversion_position(close: pd.Series) -> pd.Series:
    """True on each date the sleeve holds the symbol after that day's close."""
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


def month_end_flags(index: pd.DatetimeIndex, last_is_month_end: bool) -> pd.Series:
    """
    Flag the last trading day of each month in `index`.

    The final row can't be judged from the index alone (the month may not be
    over), so the caller says: the live runner asks the exchange calendar,
    the backtest passes False.
    """
    s = pd.Series(index, index=index)
    flags = s.groupby(index.to_period("M")).transform("max") == s
    flags.iloc[-1] = last_is_month_end
    return flags


def trend_holding(closes: pd.DataFrame, month_end: pd.Series) -> pd.Series:
    """Symbol the trend sleeve holds after each close (None before warm-up)."""
    px = closes[list(TREND_SYMBOLS)]
    momentum = px / px.shift(MOMENTUM_DAYS) - 1
    monthly = px[month_end]
    monthly_sma = monthly.rolling(TREND_SMA_MONTHS).mean()

    picks = pd.Series(None, index=closes.index, dtype=object)
    for date in monthly.index:
        mom = momentum.loc[date]
        sma = monthly_sma.loc[date]
        if mom.isna().any() or sma.isna().any():
            continue
        best = mom.idxmax()
        picks.loc[date] = best if px.loc[date, best] > sma[best] else CASH_SYMBOL
    return picks.ffill()


def target_weights(
    closes: pd.DataFrame,
    month_end: pd.Series,
    mr_weight: float = MR_WEIGHT,
    trend_weight: float = TREND_WEIGHT,
) -> pd.DataFrame:
    """Per-symbol portfolio weight to hold after each date's close."""
    weights = pd.DataFrame(0.0, index=closes.index, columns=universe())

    if mr_weight:
        per_symbol = mr_weight / len(MR_SYMBOLS)
        for sym in MR_SYMBOLS:
            weights[sym] += mean_reversion_position(closes[sym]) * per_symbol

    if trend_weight:
        holding = trend_holding(closes, month_end)
        for sym in set(TREND_SYMBOLS) | {CASH_SYMBOL}:
            weights[sym] += (holding == sym) * trend_weight

    return weights
