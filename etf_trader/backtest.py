"""
Daily backtest for strategy.target_weights().

Execution model mirrors live.py:
  * Signals are computed from the close of day t.
  * A symbol is traded on day t only if its target weight changed from day
    t-1, or it's a month-end (everything is re-sized to target then).
  * fill="close"     — filled at day t's close (live uses market-on-close).
  * fill="next_open" — filled at day t+1's open. Pessimistic cross-check:
    if the edge only exists at the exact close, it isn't robust.
  * Every fill pays cost_bps of notional (slippage + fees; Alpaca charges no
    commission, so this is all slippage/spread).

Idle cash earns nothing, which understates returns slightly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import strategy


@dataclass
class Result:
    equity: pd.Series
    trades: pd.DataFrame
    weights: pd.DataFrame


def run(
    opens: pd.DataFrame,
    closes: pd.DataFrame,
    weights: pd.DataFrame,
    month_end: pd.Series,
    cost_bps: float = 5.0,
    fill: str = "close",
    start_equity: float = 100_000.0,
) -> Result:
    if fill not in ("close", "next_open"):
        raise ValueError(fill)
    symbols = list(weights.columns)
    cl = closes[symbols].to_numpy()
    op = opens[symbols].to_numpy()
    w = weights.to_numpy()
    me = month_end.to_numpy()
    dates = closes.index
    cost = cost_bps / 1e4

    shares = np.zeros(len(symbols))
    cash = start_equity
    equity = np.empty(len(dates))
    pending = None  # (share deltas) to fill at next open
    trades = []

    for t in range(len(dates)):
        if pending is not None:
            px = op[t]
            for j in np.flatnonzero(pending):
                notional = pending[j] * px[j]
                cash -= notional + abs(notional) * cost
                shares[j] += pending[j]
                trades.append((dates[t], symbols[j], pending[j], px[j]))
            pending = None

        px = cl[t]
        # NaN closes (symbol not yet listed) only allowed where weight is 0
        value = cash + np.nansum(shares * px)
        equity[t] = value

        prev = w[t - 1] if t else np.zeros(len(symbols))
        changed = (w[t] != prev) | bool(me[t])
        if not changed.any() or t == len(dates) - 1 and fill == "next_open":
            continue

        target = np.where(w[t] > 0, w[t] * value / np.where(np.isnan(px), 1, px), 0.0)
        delta = np.where(changed, target - shares, 0.0)
        delta[np.abs(delta) < 1e-9] = 0.0
        if not delta.any():
            continue

        if fill == "close":
            for j in np.flatnonzero(delta):
                notional = delta[j] * px[j]
                cash -= notional + abs(notional) * cost
                shares[j] += delta[j]
                trades.append((dates[t], symbols[j], delta[j], px[j]))
            equity[t] = cash + np.nansum(shares * px)
        else:
            pending = delta

    return Result(
        equity=pd.Series(equity, index=dates, name="equity"),
        trades=pd.DataFrame(trades, columns=["date", "symbol", "shares", "price"]),
        weights=weights,
    )


def metrics(equity: pd.Series) -> dict:
    rets = equity.pct_change().dropna()
    years = (equity.index[-1] - equity.index[0]).days / 365.25
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1
    vol = rets.std() * np.sqrt(252)
    dd = equity / equity.cummax() - 1
    return {
        "CAGR": cagr,
        "Vol": vol,
        "Sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() else np.nan,
        "MaxDD": dd.min(),
        "Calmar": cagr / -dd.min() if dd.min() else np.nan,
        "Final": equity.iloc[-1],
    }


def mr_round_trips(closes: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    """Per-trade stats for the mean-reversion rule, entry/exit at the close."""
    rows = []
    for sym in strategy.MR_SYMBOLS:
        c = closes[sym].dropna()
        pos = strategy.mean_reversion_position(c)
        flips = pos.astype(int).diff().fillna(pos.astype(int))
        entries = c.index[flips == 1]
        exits = c.index[flips == -1]
        for i, entry in enumerate(entries):
            later = exits[exits > entry]
            if not len(later):
                break  # still open
            exit_ = later[0]
            ret = c[exit_] / c[entry] - 1 - 2 * cost_bps / 1e4
            rows.append((sym, entry, exit_, len(c.loc[entry:exit_]) - 1, ret))
    return pd.DataFrame(rows, columns=["symbol", "entry", "exit", "days", "ret"])
