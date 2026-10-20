"""
Does the dip-buy boost work on QQQ, and does adding a QQQ sleeve help?

Candidates were fixed BEFORE running this, and every one is reported:

  SPY B&H          benchmark the live bot has to beat
  QQQ B&H          benchmark for any QQQ variant
  50/50 B&H        SPY + QQQ, bought once and left to drift
  SPY boost        the live bot (SPY, rotate to SSO on the dip rule)
  QQQ boost        same rule, unchanged constants, on QQQ → QLD (2x Nasdaq-100)
  50/50 boost      half in each sleeve, each sleeve boosted by its own signal

The rule was published on the S&P in 2008, so QQQ is a genuine
out-of-sample market for it, and 2009+ is out-of-sample in time.

QLD only exists from June 2006. Before that, a synthetic 2x series is built
from QQQ exactly as history.synthetic_2x does for SPY (QLD's expense ratio,
same financing spread), and checked against real QLD where they overlap.
Synthetic history starts in 1999, so it covers the 2000–02 Nasdaq crash.

A caveat that no backtest fixes: choosing QQQ *today* uses hindsight. It
beat SPY by a wide margin after 2009 because tech led the market, which
isn't an edge and isn't guaranteed to repeat. The boost's own edge is
measured against the index it boosts (QQQ boost vs QQQ B&H).

    docker run --rm -v "$PWD":/app etf-trader-dev python research/qqq_study.py
"""

import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from etf_trader import backtest, history, strategy  # noqa: E402

COST_BPS = 5.0
QLD_EXPENSE_RATIO = 0.0095
CACHE = Path(__file__).resolve().parent.parent / "data" / "qqq_closes.csv"


def load() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE, index_col=0, parse_dates=True)
    raw = yf.download(["SPY", "SSO", "QQQ", "QLD", "^IRX"], start="1993-01-01",
                      auto_adjust=True, progress=False, threads=False)["Close"]
    raw.index = raw.index.tz_localize(None)
    raw = raw.loc[raw["SPY"].notna()]
    CACHE.parent.mkdir(exist_ok=True)
    raw.to_csv(CACHE)
    return raw


def synthetic_2x(index: pd.Series, tbill_pct: pd.Series, expense: float) -> pd.Series:
    rate = (tbill_pct.reindex(index.index).ffill().fillna(0) / 100 + history.FINANCING_SPREAD) / 252
    daily = 2 * index.pct_change() - rate - expense / 252
    return (1 + daily.fillna(0)).cumprod() * 100


def boost_weights(core: pd.Series, core_sym: str, boost_sym: str) -> pd.DataFrame:
    on = strategy.dip_position(core).astype(float)
    return pd.DataFrame({core_sym: 1 - on, boost_sym: on})


def windows(core: pd.Series, lev: pd.Series) -> pd.DataFrame:
    """Same as backtest.boost_trades, for any core/2x pair."""
    pos = strategy.dip_position(core).astype(int)
    flips = pos.diff().fillna(pos)
    entries, exits = core.index[flips == 1], core.index[flips == -1]
    rows = []
    for entry in entries:
        later = exits[exits > entry]
        if not len(later):
            break
        x = later[0]
        c, l2 = core[x] / core[entry] - 1, lev[x] / lev[entry] - 1
        rows.append((entry, l2 - c - 4 * COST_BPS / 1e4))
    return pd.DataFrame(rows, columns=["entry", "edge"])


def candidates(cl: pd.DataFrame) -> dict[str, pd.DataFrame]:
    idx = cl.index
    zero = lambda: pd.DataFrame(0.0, index=idx, columns=["SPY", "SSO", "QQQ", "QLD"])
    out = {}
    w = zero(); w["SPY"] = 1.0; out["SPY B&H"] = w
    w = zero(); w["QQQ"] = 1.0; out["QQQ B&H"] = w
    w = zero(); w[["SPY", "QQQ"]] = 0.5; out["50/50 B&H"] = w
    spy_b = boost_weights(cl["SPY"], "SPY", "SSO")
    qqq_b = boost_weights(cl["QQQ"], "QQQ", "QLD")
    w = zero(); w[["SPY", "SSO"]] = spy_b; out["SPY boost"] = w
    w = zero(); w[["QQQ", "QLD"]] = qqq_b; out["QQQ boost"] = w
    w = zero(); w[["SPY", "SSO"]] = spy_b * 0.5; w[["QQQ", "QLD"]] = qqq_b * 0.5; out["50/50 boost"] = w
    return out


def table(cl, start, end=None, fill="close"):
    rows = {}
    for name, w in candidates(cl).items():
        cost = 0 if "B&H" in name else COST_BPS
        res = backtest.run(cl, cl, w, cost, fill)
        rows[name] = backtest.metrics(res.equity.loc[start:end])
    df = pd.DataFrame(rows).T
    for c in ("CAGR", "MaxDD", "Worst yr"):
        df[c] = df[c].map(lambda x: f"{x:+.1%}")
    df["Vol"] = df["Vol"].map(lambda x: f"{x:.1%}")
    for c in ("Sharpe", "Calmar"):
        df[c] = df[c].map(lambda x: f"{x:.2f}")
    return df.to_string()


def main():
    raw = load()

    # ── Synthetic QLD check ──
    syn_qld = synthetic_2x(raw["QQQ"].dropna(), raw["^IRX"], QLD_EXPENSE_RATIO)
    real = raw["QLD"].dropna()
    yrs = (real.index[-1] - real.index[0]).days / 365.25
    s = syn_qld.loc[real.index[0]:]
    print(f"QLD since {real.index[0]:%Y-%m}: real {(real.iloc[-1] / real.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr, "
          f"synthetic {(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr\n")

    # ── Real 2x ETFs, 2006+ ──
    cl = raw.loc[real.index[0]:, ["SPY", "SSO", "QQQ", "QLD"]].dropna()
    graded = cl.index[strategy.REGIME_SMA_DAYS]
    for label, start in [(f"{graded:%Y}–now", graded), ("2009–now (out-of-sample)", "2009-01-01"),
                         ("2016–now", "2016-01-01")]:
        print(f"── Real SSO/QLD, {label} ──\n{table(cl, start)}\n")
    print(f"── Real SSO/QLD, 2009–now, every trade one day late ──")
    shifted = {}
    # One-day-late variant: trade on the following close
    for name, w in candidates(cl).items():
        cost = 0 if "B&H" in name else COST_BPS
        res = backtest.run(cl, cl, w.shift(1).bfill(), cost, "close")
        shifted[name] = backtest.metrics(res.equity.loc["2009-01-01":])
    print(pd.DataFrame(shifted).T[["CAGR", "Sharpe", "MaxDD"]].map(lambda x: f"{x:+.3f}").to_string(), "\n")

    # ── Synthetic, 1999+ (covers the dot-com crash) ──
    q = raw["QQQ"].dropna()
    syn = pd.DataFrame({
        "SPY": raw["SPY"].loc[q.index], "QQQ": q,
        "SSO": history.synthetic_2x(raw["SPY"], raw["^IRX"], history.EXPENSE_RATIOS["SSO"]).loc[q.index],
        "QLD": syn_qld,
    })
    start = syn.index[strategy.REGIME_SMA_DAYS]
    print(f"── Synthetic 2x, {start:%Y}–now ──\n{table(syn, start)}\n")
    print(f"── Synthetic 2x, {start:%Y}–2008 (dot-com + GFC) ──\n{table(syn, start, '2008-12-31')}\n")

    # ── Boost windows on QQQ ──
    print("── Boost windows (edge = 2x return − 1x return − costs) ──")
    for name, core, lev in [("SPY→SSO", syn["SPY"], syn["SSO"]), ("QQQ→QLD", syn["QQQ"], syn["QLD"])]:
        bt = windows(core.loc[start:], lev.loc[start:])
        for label, sub in [("pre-2009", bt[bt.entry < "2009-01-01"]), ("2009+", bt[bt.entry >= "2009-01-01"])]:
            t = sub.edge.mean() / sub.edge.std() * len(sub) ** 0.5
            print(f"  {name} {label}: n={len(sub)}  win={(sub.edge > 0).mean():.0%}  "
                  f"avg={sub.edge.mean():+.2%}  t={t:.1f}  worst={sub.edge.min():+.2%}")

    # How often both sleeves boost together (concentration risk)
    a = strategy.dip_position(syn["SPY"].loc[start:])
    b = strategy.dip_position(syn["QQQ"].loc[start:])
    print(f"\n  Boosted days: SPY {a.mean():.0%}, QQQ {b.mean():.0%}, both at once {(a & b).mean():.0%}")


if __name__ == "__main__":
    main()
