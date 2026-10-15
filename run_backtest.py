"""
Grade the strategy on history.

    python run_backtest.py            # uses cached data/ if present
    python run_backtest.py --refresh  # re-download from Yahoo

Reports the combined portfolio, each sleeve alone, and SPY buy-and-hold,
over the full period and split at 2009 — the rules were published in
2007–2008, so 2009 onward is out-of-sample for them.
"""

import argparse

import pandas as pd

from etf_trader import backtest, history, strategy

OOS_START = "2009-01-01"
COST_BPS = 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()

    opens, closes = history.load(refresh=args.refresh)
    symbols = strategy.universe()
    opens, closes = opens[symbols], closes[symbols]

    # Start once every ETF trades; the trend sleeve then warms up on its own
    start = closes.apply(lambda s: s.first_valid_index()).max()
    opens, closes = opens.loc[start:], closes.loc[start:]
    assert not closes.isna().any().any(), "gaps in close data"

    month_end = strategy.month_end_flags(closes.index, last_is_month_end=False)

    variants = {
        "Combined 50/50": (0.5, 0.5),
        "Mean reversion only": (1.0, 0.0),
        "Trend only": (0.0, 1.0),
    }
    curves = {}
    for name, (mr, tr) in variants.items():
        w = strategy.target_weights(closes, month_end, mr_weight=mr, trend_weight=tr)
        for fill in ("close", "next_open"):
            res = backtest.run(opens, closes, w, month_end, cost_bps=args.cost_bps, fill=fill)
            curves[(name, fill)] = res
    spy = closes["SPY"] / closes["SPY"].iloc[0] * 100_000

    # Grade everyone from the date the trend sleeve is live (~1 year warm-up)
    live_from = curves[("Trend only", "close")].weights.sum(axis=1).ne(0).idxmax()
    print(f"Data {closes.index[0]:%Y-%m-%d} → {closes.index[-1]:%Y-%m-%d}; "
          f"graded from {live_from:%Y-%m-%d}; cost {args.cost_bps:g} bps/side\n")

    periods = {
        "Full": (live_from, None),
        f"In-sample (<{OOS_START[:4]})": (live_from, pd.Timestamp(OOS_START) - pd.Timedelta(days=1)),
        f"Out-of-sample ({OOS_START[:4]}+)": (pd.Timestamp(OOS_START), None),
    }
    for label, (a, b) in periods.items():
        rows = {}
        for (name, fill), res in curves.items():
            rows[f"{name} [{fill}]"] = backtest.metrics(res.equity.loc[a:b])
        rows["SPY buy & hold"] = backtest.metrics(spy.loc[a:b])
        df = pd.DataFrame(rows).T
        print(f"── {label} ──")
        print(_fmt(df.drop(columns="Final")))
        print()

    combined = curves[("Combined 50/50", "close")]
    yearly = pd.DataFrame({
        "Combined": combined.equity.loc[live_from:].resample("YE").last().pct_change(),
        "MR only": curves[("Mean reversion only", "close")].equity.loc[live_from:].resample("YE").last().pct_change(),
        "Trend only": curves[("Trend only", "close")].equity.loc[live_from:].resample("YE").last().pct_change(),
        "SPY": spy.loc[live_from:].resample("YE").last().pct_change(),
    })
    first = {c: s.loc[live_from:].resample("YE").last().iloc[0] / s.loc[live_from] - 1
             for c, s in [("Combined", combined.equity),
                          ("MR only", curves[("Mean reversion only", "close")].equity),
                          ("Trend only", curves[("Trend only", "close")].equity),
                          ("SPY", spy)]}
    yearly.iloc[0] = pd.Series(first)
    yearly.index = yearly.index.year
    print("── Calendar-year returns (close fills) ──")
    print(yearly.map(lambda x: f"{x:+.1%}").to_string())
    print()

    rt = backtest.mr_round_trips(closes.loc[live_from:], cost_bps=args.cost_bps)
    print("── Mean-reversion round trips ──")
    for label, sub in [("All", rt), ("<2009", rt[rt.entry < OOS_START]), ("2009+", rt[rt.entry >= OOS_START])]:
        if sub.empty:
            continue
        wins = sub.ret > 0
        pf = sub.ret[wins].sum() / -sub.ret[~wins].sum()
        t_stat = sub.ret.mean() / sub.ret.std() * len(sub) ** 0.5
        print(f"  {label:6s} n={len(sub):4d}  win={wins.mean():.0%}  avg={sub.ret.mean():+.2%} (t={t_stat:.1f})  "
              f"avg win={sub.ret[wins].mean():+.2%}  avg loss={sub.ret[~wins].mean():+.2%}  "
              f"worst={sub.ret.min():+.2%}  PF={pf:.2f}  avg hold={sub.days.mean():.1f}d")
    mr_w = curves[("Mean reversion only", "close")].weights.loc[live_from:, list(strategy.MR_SYMBOLS)]
    print(f"  MR sleeve holds at least one ETF {(mr_w > 0).any(axis=1).mean():.0%} of days, "
          f"average capital deployed {mr_w.sum(axis=1).mean():.0%}")

    tr = combined.trades
    tr = tr[tr.date >= live_from]
    print(f"\nCombined: {len(tr)} fills over {(closes.index[-1] - live_from).days / 365.25:.1f} years "
          f"(~{len(tr) / ((closes.index[-1] - live_from).days / 365.25):.0f}/yr)")

    holding = strategy.trend_holding(closes, month_end).loc[live_from:]
    print("Trend sleeve time in each asset: " +
          ", ".join(f"{k} {v:.0%}" for k, v in holding.value_counts(normalize=True).items()))


def _fmt(df: pd.DataFrame) -> str:
    out = df.copy()
    for c in ("CAGR", "Vol", "MaxDD"):
        out[c] = out[c].map(lambda x: f"{x:+.1%}" if c != "Vol" else f"{x:.1%}")
    for c in ("Sharpe", "Calmar"):
        out[c] = out[c].map(lambda x: f"{x:.2f}")
    return out.to_string()


if __name__ == "__main__":
    main()
