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

The bot runs as a k8s CronJob at 15:30 and 12:30 ET on weekdays. Each run:

1. Checks the Alpaca clock and calendar. It only acts when it's within
   `RUN_WINDOW_MINUTES` of the close and before the 15:50 MOC cutoff. That
   handles half days.
2. Builds SPY history from Alpaca adjusted daily bars. Today's latest IEX
   trade stands in for today's close.
3. Submits whole-share **market-on-close** orders, sells first. It sizes off
   99% of equity, leaving a 1% buffer for movement into the close.
4. Appends one JSON line per run to `/app/logs/runs.jsonl`.

There is no state file. Positions are reconciled against the target on every
run, so a missed day self-heals the next day. A retry is also safe: each run
cancels its own open orders before resubmitting.

**Requirements:**
- **A dedicated Alpaca account.** The bot refuses to trade if it sees
  positions other than SPY or SSO.
- **A margin account.** The SPY sale and SSO purchase happen in the same
  closing auction, so the buy is submitted before the sale's cash exists.
  Paper accounts are margin by default. The bot refuses to run on a cash
  account.
- `APCA_BASE_URL` points at paper unless you set `ALLOW_LIVE=true`.

### Deploy (Flux, same pattern as lowfloat_trader)

1. Create a new Alpaca paper account. Copy `k8s/secret.env.example` to
   `k8s/secret.env` and add its keys. Then run `sops -e -i k8s/secret.env`.
2. Push to GitHub. The workflow runs the tests and pushes
   `ghcr.io/parthjp80/etf_trader:latest`.
3. Add a `GitRepository` and `Kustomization` for `./k8s` in `flux-system`,
   with a deploy key and sops decryption.
4. For a first run that places no orders, set `DRY_RUN=true` and trigger it:
   `kubectl -n trading create job etf-manual --from=cronjob/etf-trader`

### Local development

```bash
docker build -t etf-trader-dev -f - . <<'EOF'
FROM python:3.12-slim
COPY requirements-dev.txt requirements.txt ./
RUN pip install -r requirements-dev.txt
WORKDIR /app
EOF
docker run --rm -v "$PWD":/app etf-trader-dev pytest -q
docker run --rm -v "$PWD":/app etf-trader-dev python run_backtest.py
```
