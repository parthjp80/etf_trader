# ETF Trader — SPY + dip-buy boost

The bot holds 100% SPY. When SPY has a sharp short-term dip inside an uptrend,
it rotates into **SSO (2x S&P 500)** at the close. It rotates back to SPY at
the close once the dip has bounced.

| | Rule (computed on SPY's adjusted daily closes) |
|---|---|
| Boost on | close > SMA(200) **and** RSI(2) < 10 |
| Boost off | close > SMA(5) |

The dip rule comes from Connors & Alvarez, *Short Term Trading Strategies That
Work* (2008). The constants are the published ones and are not fitted to this
data. **Don't tune them against the backtest.** That's how lowfloat_trader
ended up fitting noise.

All signal logic lives in `etf_trader/strategy.py`. The backtest and the live
bot both call the same `target_weights()`.

## Backtest

Run `python run_backtest.py`. Every fill pays 5 bps.

**Real SSO** (2006-06 onward):

| Period | Boost CAGR | Sharpe | Max DD | SPY CAGR | Sharpe | Max DD |
|---|---|---|---|---|---|---|
| 2007–2026 | +12.8% | 0.65 | −55.6% | +11.0% | 0.63 | −55.2% |
| **2009–2026 (out-of-sample)** | **+16.8%** | **0.85** | **−36.9%** | +14.7% | 0.86 | −33.7% |
| 2016–2026 | +17.7% | 0.90 | −36.9% | +15.1% | 0.89 | −33.7% |

**Synthetic SSO** (1994 onward). The synthetic series is 2x SPY's daily return
minus financing and fees. Since 2006 it tracks real SSO within 0.15%/yr.

| Period | Boost CAGR | Max DD | SPY CAGR | Max DD |
|---|---|---|---|---|
| 1994–2026 | +14.1% | −55.6% | +10.9% | −55.2% |
| 1994–2026, trades one day late | +12.9% | −57.0% | | |

### What these numbers mean

- **It beat SPY by about 2–3%/yr in every period, with almost the same crash
  risk.** It holds SPY through bear markets, so 2008 was −37% for both. It is
  not downside protection.
- **Since 2009, about half of the gain is just leverage.** The bot averages
  1.11x exposure. Holding a constant 1.11x of SPY earned +15.9%/yr against
  the boost's +17.1%. So the timing edge after publication is roughly
  1.2–1.5%/yr, not 2.4%.
- **The per-trade edge has weakened since the rule was published.**

  | Boost windows | Count | Avg edge vs SPY | t-stat |
  |---|---|---|---|
  | 1994–2008 | 110 | +0.56% | 4.1 |
  | 2009–now | 149 | +0.29% | 2.2 |

  Expect live results to be lower again.
- **It lags SPY in about a third of years.** The worst year was 2011 at −11%
  vs SPY. 2018 was −8.5%.
- **It boosts about 11% of days**, roughly 8 times a year at 3–4 days each.
  That's about 32 fills a year.
- **Taxes:** every rotation sells SPY and realizes the gain. In a taxable
  account that drag could wipe out the edge. It's best run in an IRA or other
  tax-advantaged account.

`research/leverage_study.py` compares this against the other candidates
tested: 2x/3x trend timing, and 2x held always. All five were fixed before
any of them was run.

## Running it

The bot runs as a k8s CronJob at 15:45 and 12:45 ET on weekdays. Each run:

1. Checks the Alpaca clock and calendar. It only acts when it's within
   `RUN_WINDOW_MINUTES` (20) of the close and at least 2 minutes before it. That
   handles half days.
2. Builds SPY history from Alpaca adjusted daily bars. Today's latest IEX
   trade stands in for today's close.
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
  positions other than SPY or SSO.
- **A margin account.** The SPY sale and SSO purchase go in back to back,
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
