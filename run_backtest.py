"""
Grade SPY + dip-buy boost against SPY buy-and-hold.

    python run_backtest.py            # uses cached data/ if present
    python run_backtest.py --refresh  # re-download from Yahoo

Two histories:
  * Real SSO, 2006-06 onward. Fills at the close and at the next open.
  * Synthetic SSO, 1994 onward (see history.synthetic_2x). Close fills only.

The dip rule was published in 2008, so 2009 onward is out-of-sample.
"""

import argparse

import pandas as pd

from etf_trader import backtest, history, strategy

COST_BPS = 5.0
CORE, BOOST = strategy.CORE_SYMBOL, strategy.BOOST_SYMBOL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()

    opens, closes = history.load(refresh=args.refresh)
    syms = strategy.universe()

    # ── Synthetic SSO check ──
    syn = history.synthetic_2x(closes[CORE], closes["^IRX"])
    real = closes[BOOST].dropna()
    yrs = (real.index[-1] - real.index[0]).days / 365.25
    s = syn.loc[real.index[0]:]
    print(f"SSO since {real.index[0]:%Y-%m}: real {(real.iloc[-1] / real.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr, "
          f"synthetic {(s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr")
    print(f"Cost {args.cost_bps:g} bps per fill\n")

    # ── Real SSO, 2006+ ──
    real_cl = closes.loc[real.index[0]:, syms]
    real_op = opens.loc[real.index[0]:, syms]
    w_real = strategy.target_weights(real_cl[CORE])
    runs = {
        "Boost [close]": backtest.run(real_op, real_cl, w_real, args.cost_bps, "close"),
        "Boost [next open]": backtest.run(real_op, real_cl, w_real, args.cost_bps, "next_open"),
        "SPY buy & hold": backtest.run(real_op, real_cl, w_real.assign(SPY=1.0, SSO=0.0), 0, "close"),
    }
    # Grade from when the 200-day SMA exists, so both sides start together
    graded = real_cl.index[strategy.REGIME_SMA_DAYS]
    report("Real SSO", runs, {"2007–now": (graded, None),
                              "2009–now (out-of-sample)": ("2009-01-01", None),
                              "2016–now": ("2016-01-01", None)})

    # ── Synthetic SSO, 1994+ ──
    syn_cl = pd.DataFrame({CORE: closes[CORE], BOOST: syn})
    w_syn = strategy.target_weights(syn_cl[CORE])
    syn_runs = {
        "Boost [close]": backtest.run(syn_cl, syn_cl, w_syn, args.cost_bps, "close"),
        "Boost [+1 day late]": backtest.run(syn_cl, syn_cl, w_syn.shift(1).fillna({CORE: 1.0, BOOST: 0.0}),
                                            args.cost_bps, "close"),
        "SPY buy & hold": backtest.run(syn_cl, syn_cl, w_syn.assign(SPY=1.0, SSO=0.0), 0, "close"),
    }
    start = syn_cl.index[strategy.REGIME_SMA_DAYS]
    report("Synthetic SSO", syn_runs, {"1994–now": (start, None),
                                       "1994–2008 (in-sample)": (start, "2008-12-31")})

    yearly = pd.DataFrame({
        "Boost": syn_runs["Boost [close]"].equity.loc[start:].resample("YE").last().pct_change(),
        "SPY": syn_runs["SPY buy & hold"].equity.loc[start:].resample("YE").last().pct_change(),
    }).dropna()
    yearly["Diff"] = yearly["Boost"] - yearly["SPY"]
    yearly.index = yearly.index.year
    print("── Calendar years (synthetic SSO before mid-2006) ──")
    print(yearly.map(lambda x: f"{x:+.1%}").to_string())
    print(f"Boost beat SPY in {(yearly['Diff'] > 0).mean():.0%} of years; "
          f"worst year vs SPY {yearly['Diff'].min():+.1%}, best {yearly['Diff'].max():+.1%}\n")

    bt = backtest.boost_trades(syn_cl.loc[start:], args.cost_bps)
    print("── Boost windows (edge = SSO return − SPY return − costs) ──")
    for label, sub in [("1994–2008", bt[bt.entry < "2009-01-01"]), ("2009–now", bt[bt.entry >= "2009-01-01"])]:
        wins = sub.edge > 0
        t = sub.edge.mean() / sub.edge.std() * len(sub) ** 0.5
        print(f"  {label}: n={len(sub)}  win={wins.mean():.0%}  avg edge={sub.edge.mean():+.2%} (t={t:.1f})  "
              f"worst={sub.edge.min():+.2%}  avg hold={sub.days.mean():.1f}d")
    on = w_syn.loc[start:, BOOST].mean()
    print(f"  Boosted {on:.0%} of days; ~{len(bt) / ((syn_cl.index[-1] - start).days / 365.25):.0f} windows/yr "
          f"(4 fills each)")


def report(title, runs, periods):
    for label, (start, end) in periods.items():
        rows = {name: backtest.metrics(res.equity.loc[start:end]) for name, res in runs.items()}
        df = pd.DataFrame(rows).T
        for c in ("CAGR", "MaxDD", "Worst yr"):
            df[c] = df[c].map(lambda x: f"{x:+.1%}")
        df["Vol"] = df["Vol"].map(lambda x: f"{x:.1%}")
        for c in ("Sharpe", "Calmar"):
            df[c] = df[c].map(lambda x: f"{x:.2f}")
        print(f"── {title}: {label} ──")
        print(df.to_string())
        print()


if __name__ == "__main__":
    main()
