"""
Daily backtest for strategy.target_weights().

Execution model mirrors live.py:
  * Signals are computed from the close of day t.
  * A symbol is traded on day t only if its target weight changed from
    day t-1, or its holding has drifted more than rebalance_tolerance off
    target (strategy.REBALANCE_TOLERANCE, as live). Otherwise holdings
    drift with the market.
  * fill="close"     — filled at day t's close (live fills ~15 min before it).
  * fill="next_open" — filled at day t+1's open. A pessimistic cross-check:
    if the edge only exists at the exact close, it isn't robust.
  * Every fill pays cost_bps of notional. Alpaca charges no commission, so
    this cost is all slippage and spread.
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
    cost_bps: float = 5.0,
    fill: str = "close",
    start_equity: float = 100_000.0,
    rebalance_tolerance: float | None = strategy.REBALANCE_TOLERANCE,
) -> Result:
    if fill not in ("close", "next_open"):
        raise ValueError(fill)
    symbols = list(weights.columns)
    cl = closes[symbols].to_numpy()
    op = opens[symbols].to_numpy()
    w = weights.to_numpy()
    dates = closes.index
    cost = cost_bps / 1e4

    shares = np.zeros(len(symbols))
    cash = start_equity
    equity = np.empty(len(dates))
    pending = None  # share deltas to fill at the next open
    trades = []

    def fill_orders(delta, px, when):
        nonlocal cash
        for j in np.flatnonzero(delta):
            notional = delta[j] * px[j]
            cash -= notional + abs(notional) * cost
            shares[j] += delta[j]
            trades.append((when, symbols[j], delta[j], px[j]))

    for t in range(len(dates)):
        if pending is not None:
            fill_orders(pending, op[t], dates[t])
            pending = None

        px = cl[t]
        value = cash + shares @ px
        equity[t] = value

        prev = w[t - 1] if t else np.zeros(len(symbols))
        target = w[t] * value / px
        changed = w[t] != prev
        if rebalance_tolerance is not None:
            changed |= (shares != 0) & (np.abs(target - shares) > rebalance_tolerance * target)
        if not changed.any() or (fill == "next_open" and t == len(dates) - 1):
            continue

        delta = np.where(changed, target - shares, 0.0)
        delta[np.abs(delta) < 1e-9] = 0.0
        if not delta.any():
            continue

        if fill == "close":
            fill_orders(delta, px, dates[t])
            equity[t] = cash + shares @ px
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
    dd = equity / equity.cummax() - 1
    return {
        "CAGR": cagr,
        "Vol": rets.std() * np.sqrt(252),
        "Sharpe": rets.mean() / rets.std() * np.sqrt(252) if rets.std() else np.nan,
        "MaxDD": dd.min(),
        "Calmar": cagr / -dd.min() if dd.min() else np.nan,
        "Worst yr": equity.resample("YE").last().pct_change().min(),
    }


def boost_trades(core: pd.Series, lev: pd.Series, cost_bps: float = 5.0) -> pd.DataFrame:
    """
    One row per boost window of one sleeve: what the extra 1x earned.

    The boost's payoff is the 2x ETF's return minus the return of the core
    it replaced. Each window costs two round trips (core→2x, 2x→core), i.e.
    four fills.
    """
    pos = strategy.dip_position(core).astype(int)
    flips = pos.diff().fillna(pos)
    entries, exits = core.index[flips == 1], core.index[flips == -1]
    rows = []
    for entry in entries:
        later = exits[exits > entry]
        if not len(later):
            break  # still open
        exit_ = later[0]
        core_ret = core[exit_] / core[entry] - 1
        lev_ret = lev[exit_] / lev[entry] - 1
        rows.append((entry, exit_, len(core.loc[entry:exit_]) - 1, core_ret,
                     lev_ret - core_ret - 4 * cost_bps / 1e4))
    return pd.DataFrame(rows, columns=["entry", "exit", "days", "core_ret", "edge"])
