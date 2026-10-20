"""
Grade 50/50 SPY + QQQ with dip-buy boosts against the obvious alternatives.

    python run_backtest.py            # uses cached data/ if present
    python run_backtest.py --refresh  # re-download from Yahoo

Two histories:
  * Real SSO/QLD, 2006-06 onward. Fills at the close and at the next open.
  * Synthetic SSO/QLD, 1999 onward (see history.synthetic_2x), covering the
    2000–02 Nasdaq crash. Close fills, plus a one-day-late variant.

Benchmarks: SPY held, the same 50/50 SPY/QQQ mix held (with the same 10%
rebalance band, no boosts), and the previous SPY-only boost. The dip rule
was published in 2008, so 2009 onward is out-of-sample.
"""

import argparse

import pandas as pd

from etf_trader import backtest, history, strategy

COST_BPS = 5.0
SYMS = strategy.universe()
CORES = strategy.cores()


def flat(index, **weights) -> pd.DataFrame:
    w = pd.DataFrame(0.0, index=index, columns=SYMS)
    for sym, x in weights.items():
        w[sym] = x
    return w


def spy_only_boost(cl: pd.DataFrame) -> pd.DataFrame:
    on = strategy.dip_position(cl["SPY"]).astype(float)
    return flat(cl.index, SPY=1 - on, SSO=on)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()
    cost = args.cost_bps

    opens, closes = history.load(refresh=args.refresh)

    # ── Synthetic 2x checks ──
    syn = {}
    for s in strategy.SLEEVES:
        syn[s.boost] = history.synthetic_2x(closes[s.core].dropna(), closes["^IRX"],
                                            history.EXPENSE_RATIOS[s.boost])
        real = closes[s.boost].dropna()
        yrs = (real.index[-1] - real.index[0]).days / 365.25
        x = syn[s.boost].loc[real.index[0]:]
        print(f"{s.boost} since {real.index[0]:%Y-%m}: real {(real.iloc[-1] / real.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr, "
              f"synthetic {(x.iloc[-1] / x.iloc[0]) ** (1 / yrs) - 1:+.2%}/yr")
    print(f"Cost {cost:g} bps per fill\n")

    # ── Real SSO/QLD, 2006+ ──
    real_cl = closes[SYMS].dropna()
    real_op = opens.loc[real_cl.index, SYMS]
    w = strategy.target_weights(real_cl[CORES])
    runs = {
        "50/50 boost [close]": backtest.run(real_op, real_cl, w, cost, "close"),
        "50/50 boost [next open]": backtest.run(real_op, real_cl, w, cost, "next_open"),
        "50/50 held": backtest.run(real_op, real_cl, flat(real_cl.index, SPY=0.5, QQQ=0.5), 0, "close"),
        "SPY boost (previous bot)": backtest.run(real_op, real_cl, spy_only_boost(real_cl), cost, "close"),
        "SPY held": backtest.run(real_op, real_cl, flat(real_cl.index, SPY=1.0), 0, "close"),
    }
    graded = real_cl.index[strategy.REGIME_SMA_DAYS]
    report("Real SSO/QLD", runs, {f"{graded:%Y}–now": (graded, None),
                                  "2009–now (out-of-sample)": ("2009-01-01", None),
                                  "2016–now": ("2016-01-01", None)})

    # ── Synthetic SSO/QLD, 1999+ ──
    q = closes["QQQ"].dropna()
    syn_cl = pd.DataFrame({"SPY": closes["SPY"], "SSO": syn["SSO"], "QQQ": q, "QLD": syn["QLD"]}).loc[q.index, SYMS]
    w_syn = strategy.target_weights(syn_cl[CORES])
    late = w_syn.shift(1)
    late.iloc[0] = w_syn.iloc[0]
    syn_runs = {
        "50/50 boost [close]": backtest.run(syn_cl, syn_cl, w_syn, cost, "close"),
        "50/50 boost [+1 day late]": backtest.run(syn_cl, syn_cl, late, cost, "close"),
        "50/50 held": backtest.run(syn_cl, syn_cl, flat(syn_cl.index, SPY=0.5, QQQ=0.5), 0, "close"),
        "SPY boost (previous bot)": backtest.run(syn_cl, syn_cl, spy_only_boost(syn_cl), cost, "close"),
        "SPY held": backtest.run(syn_cl, syn_cl, flat(syn_cl.index, SPY=1.0), 0, "close"),
    }
    start = syn_cl.index[strategy.REGIME_SMA_DAYS]
    report("Synthetic SSO/QLD", syn_runs, {f"{start:%Y}–now": (start, None),
                                           f"{start:%Y}–2008 (dot-com + GFC)": (start, "2008-12-31")})

    yearly = pd.DataFrame({
        name: syn_runs[key].equity.loc[start:].resample("YE").last().pct_change()
        for name, key in [("Boost", "50/50 boost [close]"), ("50/50 held", "50/50 held"), ("SPY", "SPY held")]
    }).dropna()
    yearly["vs 50/50"] = yearly["Boost"] - yearly["50/50 held"]
    yearly["vs SPY"] = yearly["Boost"] - yearly["SPY"]
    yearly.index = yearly.index.year
    print("── Calendar years (synthetic 2x before mid-2006) ──")
    print(yearly.map(lambda x: f"{x:+.1%}").to_string())
    print(f"Beat 50/50 held in {(yearly['vs 50/50'] > 0).mean():.0%} of years, SPY in {(yearly['vs SPY'] > 0).mean():.0%}; "
          f"worst year vs SPY {yearly['vs SPY'].min():+.1%}\n")

    print("── Boost windows (edge = 2x return − 1x return − costs) ──")
    for s in strategy.SLEEVES:
        bt = backtest.boost_trades(syn_cl[s.core].loc[start:], syn_cl[s.boost].loc[start:], cost)
        for label, sub in [("1999–2008", bt[bt.entry < "2009-01-01"]), ("2009–now", bt[bt.entry >= "2009-01-01"])]:
            t = sub.edge.mean() / sub.edge.std() * len(sub) ** 0.5
            print(f"  {s.core}→{s.boost} {label}: n={len(sub)}  win={(sub.edge > 0).mean():.0%}  "
                  f"avg edge={sub.edge.mean():+.2%} (t={t:.1f})  worst={sub.edge.min():+.2%}  "
                  f"avg hold={sub.days.mean():.1f}d")
    on = {s.core: strategy.dip_position(syn_cl[s.core].loc[start:]) for s in strategy.SLEEVES}
    both = on["SPY"] & on["QQQ"]
    lev = 1 + sum(w_syn.loc[start:, s.boost] for s in strategy.SLEEVES)
    print(f"  Boosted days: SPY {on['SPY'].mean():.0%}, QQQ {on['QQQ'].mean():.0%}, "
          f"both (whole account 2x) {both.mean():.0%}; average exposure {lev.mean():.2f}x")
    trades = syn_runs["50/50 boost [close]"].trades
    trades = trades[trades.date >= start]
    print(f"  ~{len(trades) / ((syn_cl.index[-1] - start).days / 365.25):.0f} fills/yr")


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
