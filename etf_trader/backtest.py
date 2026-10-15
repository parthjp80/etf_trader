"""
Daily backtest for strategy.target_weights().

Execution model mirrors live.py:
  * Signals are computed from the close of day t.
  * A symbol is traded on day t only if its target weight changed from
    day t-1. Between changes, holdings drift with the market.
  * fill="close"     — filled at day t's close (live uses market-on-close).
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
        changed = w[t] != prev
        if not changed.any() or (fill == "next_open" and t == len(dates) - 1):
            continue

        target = w[t] * value / px
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


def boost_trades(closes: pd.DataFrame, cost_bps: float = 5.0) -> pd.DataFrame:
    """
    One row per boost window: what the extra 1x of exposure earned.

    The boost's payoff is the SSO return minus the SPY return it replaced.
    Each window costs two round trips (SPY→SSO, SSO→SPY), i.e. four fills.
    """
    spy, sso = closes[strategy.CORE_SYMBOL], closes[strategy.BOOST_SYMBOL]
    pos = strategy.dip_position(spy).astype(int)
    flips = pos.diff().fillna(pos)
    entries, exits = spy.index[flips == 1], spy.index[flips == -1]
    rows = []
    for entry in entries:
        later = exits[exits > entry]
        if not len(later):
            break  # still open
        exit_ = later[0]
        spy_ret = spy[exit_] / spy[entry] - 1
        sso_ret = sso[exit_] / sso[entry] - 1
        rows.append((entry, exit_, len(spy.loc[entry:exit_]) - 1, spy_ret,
                     sso_ret - spy_ret - 4 * cost_bps / 1e4))
    return pd.DataFrame(rows, columns=["entry", "exit", "days", "spy_ret", "edge"])
