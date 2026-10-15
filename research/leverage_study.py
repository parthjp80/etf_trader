"""
Can a rules-based SPY strategy beat SPY buy-and-hold with acceptable risk?

Candidates were fixed BEFORE running this, and every one is reported, so
the winner isn't hand-picked from a large search:

  SPY        buy & hold (benchmark)
  LRS-2x     Gayed & Bilello, "Leverage for the Long Run" (2016): hold 2x S&P
             when SPY close > SMA(200), else T-bills (SHY).
  LRS-3x     same with 3x.
  CORE+MR    100% SPY always; switch to 2x S&P while the RSI(2) mean-reversion
             rule (strategy.mean_reversion_position) is long.
  2x B&H     2x S&P held always — shows what leverage alone does.

Leveraged ETFs (SSO 2006, UPRO 2009) are too young for 1993+ history, so
they're simulated daily: L * SPY return - (L-1) * T-bill rate - expense
ratio. The simulation is checked against real SSO/UPRO where they overlap.
Robustness: every rotation is also run with a one-day trading delay.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etf_trader import backtest, strategy  # noqa: E402

COST_BPS = 5.0
EXPENSE = {2: 0.0089, 3: 0.0091}
# Swap financing costs more than the T-bill rate. Calibrated so the simulation
# matches real SSO/UPRO (both overstate by ~0.8%/yr per borrowed unit without it).
FINANCING_SPREAD = 0.008


def load():
    raw = yf.download(["SPY", "SHY", "SSO", "UPRO", "^IRX"], start="1993-01-01",
                      auto_adjust=True, progress=False, threads=False)["Close"]
    raw.index = raw.index.tz_localize(None)
    return raw


def synthetic(spy, tbill_annual, lev):
    r = spy.pct_change()
    daily_fin = (tbill_annual + FINANCING_SPREAD) / 252
    syn = lev * r - (lev - 1) * daily_fin - EXPENSE[lev] / 252
    return (1 + syn.fillna(0)).cumprod() * 100


def main():
    raw = load()
    spy = raw["SPY"].dropna()
    tbill = (raw["^IRX"].reindex(spy.index).ffill() / 100).fillna(0)
    # Cash: real SHY from 2002, before that T-bill accrual
    cash_syn = (1 + tbill / 252).cumprod() * 100
    shy = raw["SHY"].reindex(spy.index)
    first = shy.first_valid_index()
    cash = cash_syn.copy()
    cash.loc[first:] = shy.loc[first:] / shy.loc[first] * cash_syn.loc[first]

    px = pd.DataFrame({"SPY": spy, "X2": synthetic(spy, tbill, 2),
                       "X3": synthetic(spy, tbill, 3), "CASH": cash}).dropna()

    print("Simulation check vs real leveraged ETFs (annualised return):")
    for real, col in (("SSO", "X2"), ("UPRO", "X3")):
        r = raw[real].dropna()
        s = px[col].loc[r.index[0]:]
        yrs = (r.index[-1] - r.index[0]).days / 365.25
        print(f"  {real}: real {(r.iloc[-1]/r.iloc[0])**(1/yrs)-1:+.2%}  "
              f"simulated {(s.iloc[-1]/s.iloc[0])**(1/yrs)-1:+.2%}  since {r.index[0]:%Y-%m}")

    above = px["SPY"] > px["SPY"].rolling(200).mean()
    mr = strategy.mean_reversion_position(px["SPY"])

    def w(**cols):
        out = pd.DataFrame(0.0, index=px.index, columns=px.columns)
        for c, v in cols.items():
            out[c] = v.astype(float) if hasattr(v, "astype") else float(v)
        return out

    cands = {
        "SPY B&H": w(SPY=1),
        "2x B&H": w(X2=1),
        "LRS-2x": w(X2=above, CASH=~above),
        "LRS-3x": w(X3=above, CASH=~above),
        "CORE+MR": w(SPY=~mr, X2=mr),
    }
    me = pd.Series(False, index=px.index)
    start = px.index[200]
    results = {}
    for name, wt in cands.items():
        for delay in (0, 1):
            wt_d = wt.shift(delay).fillna(0) if delay else wt
            if name.endswith("B&H") and delay:
                continue
            res = backtest.run(px.loc[start:], px.loc[start:], wt_d.loc[start:], me.loc[start:], cost_bps=COST_BPS)
            label = name + (" [+1d delay]" if delay else "")
            results[label] = res

    periods = {"1994–2026": (start, None), "2009–2026": ("2009-01-01", None),
               "2016–2026 (post-publication)": ("2016-06-01", None)}
    for p, (a, b) in periods.items():
        rows = {}
        for k, res in results.items():
            e = res.equity.loc[a:b]
            m = backtest.metrics(e)
            yearly = e.resample("YE").last().pct_change().dropna()
            m["Worst yr"] = yearly.min()
            m["Trades/yr"] = len(res.trades[res.trades.date >= pd.Timestamp(a)]) / ((e.index[-1]-e.index[0]).days/365.25)
            m["$10k→"] = 10_000 * e.iloc[-1] / e.iloc[0]
            rows[k] = m
        df = pd.DataFrame(rows).T.drop(columns="Final")
        print(f"\n── {p} ──")
        fmt = {"CAGR": "{:+.1%}", "Vol": "{:.1%}", "Sharpe": "{:.2f}", "MaxDD": "{:+.1%}",
               "Calmar": "{:.2f}", "Worst yr": "{:+.1%}", "Trades/yr": "{:.1f}", "$10k→": "${:,.0f}"}
        print(df.apply(lambda col: col.map(fmt[col.name].format)).to_string())

    # Year-by-year for the main candidates
    keys = ["SPY B&H", "LRS-2x", "LRS-3x", "CORE+MR"]
    yr = pd.DataFrame({k: results[k].equity.resample("YE").last().pct_change() for k in keys}).dropna()
    yr.index = yr.index.year
    print("\n── Calendar years ──")
    print(yr.map(lambda x: f"{x:+.0%}").to_string())

    # Rolling 5-year: how often did each beat SPY?
    print("\n── Share of rolling 5-year windows beating SPY B&H ──")
    base = results["SPY B&H"].equity
    for k in keys[1:]:
        e = results[k].equity
        r5 = (e / e.shift(1260)) / (base / base.shift(1260))
        print(f"  {k}: {(r5.dropna() > 1).mean():.0%}")


if __name__ == "__main__":
    main()
