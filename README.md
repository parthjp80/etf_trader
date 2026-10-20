# ETF Trader — 50/50 SPY + QQQ with dip-buy boosts

The account is split into two equal sleeves: **SPY** and **QQQ**. Each
sleeve has its own dip rule, computed on its own index. When a sharp
short-term dip hits inside an uptrend, that sleeve rotates into its 2x ETF
(**SSO** for SPY, **QLD** for QQQ). It rotates back once the dip has bounced.
Either sleeve, both or neither can be boosted on a given day.

| | Rule (each sleeve, on its own index's adjusted daily closes) |
|---|---|
| Boost on | close > SMA(200) **and** RSI(2) < 10 |
| Boost off | close > SMA(5) |

The dip rule comes from Connors & Alvarez, *Short Term Trading Strategies That
Work* (2008). The constants are the published ones and are not fitted to this
data. **Don't tune them against the backtest.** That's how lowfloat_trader
ended up fitting noise.

Between signal changes the sleeves drift with the market. A position more
than 10% off its target size is re-sized, so the mix is pulled back to
50/50 once it drifts past about 55/45. The backtest and the live bot apply
the same rule.

All signal logic lives in `etf_trader/strategy.py`. The backtest and the live
bot both call the same `target_weights()`.

Until 2026-10-21 the bot ran a SPY-only version (100% SPY, boosted into SSO).
`research/qqq_study.py` is where the 50/50 version was chosen, from six
candidates that were all fixed before any of them was run.

## Backtest

Run `python run_backtest.py`. Every fill pays 5 bps.

**Real SSO/QLD, 2009–2026 (out-of-sample):**

| | CAGR | Sharpe | Max DD | Worst yr |
|---|---|---|---|---|
| **50/50 boost** | **+19.7%** | **0.91** | **−33.6%** | −28.4% |
| 50/50 boost, filled at next open | +19.2% | 0.90 | −33.8% | −28.2% |
| 50/50 SPY/QQQ held | +17.6% | 0.95 | −30.7% | −25.4% |
| SPY boost (previous bot) | +16.7% | 0.84 | −36.9% | −21.2% |
| SPY held | +14.6% | 0.86 | −33.7% | −18.2% |

**Synthetic SSO/QLD, 1999–2026.** The synthetic series is 2x the index's
daily return minus financing and fees. Since 2006 it tracks real SSO and QLD
within about 0.2%/yr. It covers the 2000–02 Nasdaq crash.

| | CAGR | Max DD |
|---|---|---|
| **50/50 boost** | **+10.8%** | **−70.1%** |
| 50/50 boost, trades one day late | +10.1% | −70.4% |
| 50/50 SPY/QQQ held | +8.8% | −68.8% |
| SPY boost (previous bot) | +10.2% | −55.6% |
| SPY held | +8.3% | −55.2% |

### What these numbers mean

- **Most of the gain over SPY comes from holding QQQ, not from the boost.**
  QQQ beat SPY by a wide margin after 2009 because tech led the market.
  Choosing it now uses hindsight, and that lead isn't guaranteed to last.
- **The boost adds about 2%/yr over the same mix held, with a slightly lower
  Sharpe.** Like the SPY-only version, much of that is extra leverage: the
  bot averages 1.11x exposure.
- **The crash risk is higher than SPY's.** From 1999 to 2008 this lost
  5.7%/yr, with a −70% drawdown, against −51% for SPY. A dot-com-style tech
  crash would hurt this far more than the SPY-only bot.
- **The rule works on QQQ as well as on SPY.** It was published on the S&P
  and never fitted to QQQ, and its per-window edge since 2009 matches:

  | Boost windows, 2009–now | Count | Avg edge vs core | t-stat |
  |---|---|---|---|
  | SPY → SSO | 150 | +0.29% | 2.1 |
  | QQQ → QLD | 156 | +0.29% | 2.0 |

  Expect live results to be lower.
- **The two boosts mostly fire together.** SPY boosts about 10% of days and
  QQQ about 11%; on 7% of days both are boosted and the whole account is at
  2x.
- **It lags in about a third of years.** It beat the 50/50 mix held in 67% of
  years and SPY in 63%. The worst years vs SPY were 2022 (−10.3%), 2011
  (−9.4%) and 2000 (−9.4%).
- **About 62 fills a year**, counting both sleeves and rebalances.
- **Taxes:** every rotation sells the core and realizes the gain. In a
  taxable account that drag could wipe out the edge. It's best run in an IRA
  or other tax-advantaged account.

`research/leverage_study.py` compares the original SPY version against 2x/3x
trend timing and 2x held always.

## Running it

The bot runs as a k8s CronJob at 15:45 and 12:45 ET on weekdays. Each run:

1. Checks the Alpaca clock and calendar. It only acts when it's within
   `RUN_WINDOW_MINUTES` (20) of the close and at least 2 minutes before it. That
   handles half days.
2. Builds SPY and QQQ history from Alpaca adjusted daily bars. Today's
   latest IEX trade stands in for today's close.
3. Submits whole-share **market** orders, sells first, which fill right
   away. It sizes off 99% of equity, leaving a 1% buffer. It doesn't use
   market-on-close orders: Alpaca paper fills them only partly at random
   (608 of 1,113 and 0 of 497 in the first two runs), and live accounts
   only get them with Alpaca's elite order routing. Filling 15 minutes
   early costs little next to the backtest's one-day-late variant (−1.2%/yr).
4. Appends one JSON line per run to `/app/logs/runs.jsonl`.

There is no state file. Positions are reconciled against the target on every
run, so a missed day self-heals the next day, and so does a partial fill
more than 10% short of target. A retry is also safe: it cancels anything
left unfilled and re-plans from the positions that actually filled.

**Requirements:**
- **A dedicated Alpaca account.** The bot refuses to trade if it sees
  positions other than SPY, SSO, QQQ or QLD.
- **A margin account.** A rotation's sale and purchase go in back to back,
  so the buy is submitted before the sale's cash settles.
  Paper accounts are margin by default. The bot refuses to run on a cash
  account.
- `APCA_BASE_URL` points at paper unless you set `ALLOW_LIVE=true`.

### Deploy (Flux, same pattern as lowfloat_trader)

Already wired up:
- `flux/etf-trader.yaml` defines the `GitRepository` and `Kustomization` in
  `flux-system`. They're applied directly to the cluster and track `./k8s`
  on `main`.
- The repo has a read-only deploy key, stored as the `etf-trader-deploy-key`
  secret in `flux-system`.
- `k8s/secret.env` is sops-encrypted with the same age key as the cluster's
  `sops-age` secret, so Flux decrypts it on reconcile.
- The bot shares the `trading` namespace and uses an `nfs-csi` PVC
  (`etf-trader-logs`).

To go live on paper:

1. **Make the image public.** Go to
   github.com/users/parthjp80/packages/container/etf_trader/settings →
   Change visibility. The image has no secrets in it.
2. **Add keys** for a *dedicated* Alpaca paper account. Run
   `sops k8s/secret.env` and replace the `REPLACE_ME` placeholders.
3. **Dry run.** Push, then trigger a manual run inside the last 40 minutes
   before the close:
   ```bash
   kubectl -n trading create job etf-manual --from=cronjob/etf-trader
   kubectl -n trading logs -f job/etf-manual
   ```
   `DRY_RUN=true` in `k8s/configmap.env` logs the plan without placing
   orders.
4. **Turn it on.** Set `DRY_RUN=false` in `k8s/configmap.env` and
   `suspend: false` in `k8s/cronjob.yaml`, then push.

### End-of-day email

`k8s/cronjob-report.yaml` runs `python -m etf_trader.report` at 16:20 ET on
weekdays (holidays skip themselves). The email has the day's P&L, the return
since the first trade against SPY over the same span, holdings, today's
orders with fill status, and the signal: what would flip it at the next
close. Partial fills, failed runs and a missing run get a warning banner and
a ⚠️ in the subject. It uses the same Gmail credentials as lowfloat_trader
(`EMAIL_FROM`, `EMAIL_TO`, `EMAIL_PASSWORD` in `k8s/secret.env`).

```bash
kubectl -n trading create job etf-report-now --from=cronjob/etf-trader-report
```

Everyday operations:

```bash
kubectl -n trading get cronjob etf-trader         # schedule / last run
kubectl -n trading get jobs -l app.kubernetes.io/name=etf-trader
kubectl -n trading logs job/<job-name>            # one JSON line per run
flux reconcile kustomization etf-trader --with-source   # deploy now instead of waiting ~1m
```

Code changes deploy on `git push`. The workflow tests the code and pushes
`:latest`, and the CronJob pulls with `imagePullPolicy: Always`, so the next
run picks it up with no restart needed.

### Local development

```bash
docker build -t etf-trader-dev -f - . <<'EOF'
FROM python:3.12-slim
COPY requirements-dev.txt requirements.txt ./
RUN pip install -r requirements-dev.txt
WORKDIR /app
EOF
docker run --rm -v "$PWD":/app etf-trader-dev python -m pytest -q
docker run --rm -v "$PWD":/app etf-trader-dev python run_backtest.py
```
