# ETF Trader

A daily, end-of-day ETF strategy for Alpaca. It has two sleeves, each 50% of
equity. Both use rules that were published before 2009 and are not fitted to
this data.

| Sleeve | Universe | Rule |
|---|---|---|
| Mean reversion | SPY, QQQ (25% each) | Buy at the close when close > SMA(200) and RSI(2) < 10. Sell at the close when close > SMA(5). (Connors & Alvarez, 2008) |
| Trend | SPY, EFA, TLT, GLD, cash = SHY | Each month-end, take the best 252-day performer. Hold it if it's above its 10-month SMA, otherwise hold SHY. (Antonacci 2014; Faber 2007) |

All signal logic lives in `etf_trader/strategy.py`. The backtest and the live
bot both call the same `target_weights()`, so what goes live is exactly what
was tested.

**Do not tune the constants in `strategy.py` against the backtest.** That is
how the lowfloat bot ended up fitting noise.

## Backtest results

Run with `python run_backtest.py`. The data is Yahoo adjusted daily bars,
2005-11 to 2026-10. Costs are 5 bps per side, and idle cash earns 0%.

| Period | Strategy CAGR | Sharpe | Max DD | SPY CAGR | SPY Sharpe | SPY Max DD |
|---|---|---|---|---|---|---|
| Full (2005–2026) | +6.5% | 0.71 | −14.8% | +11.1% | 0.65 | −55.2% |
| Before 2009 | +11.5% | 1.03 | −13.5% | −8.2% | −0.20 | −50.8% |
| **2009+ (out-of-sample)** | **+5.7%** | **0.66** | **−14.8%** | **+14.7%** | **0.86** | **−33.7%** |

Filling at the next day's open instead of the close costs about 0.4–0.5%/yr.
The result doesn't depend on hitting the exact close.

What the numbers say:

- **Since 2009 it lost to plain SPY buy-and-hold on both return and Sharpe.**
  What you get for that is a much smaller drawdown: −15% vs −34% to −55%.
  This is a defensive strategy, not a way to make more money than the index.
- **Mean reversion has a real per-trade edge.** It made 305 trades since 2009:
  70% won, the average was +0.47%, t ≈ 4.9, and profit factor was 2.1.
  But it only has about 12% of capital deployed on average, so it adds only
  ~3.9%/yr to the whole account. It also has no stop. Its worst single trade
  was −10.3%.
- **The trend sleeve is the weak part since 2009**: Sharpe 0.50 and a −29%
  max drawdown. It looked brilliant up to 2008 (+31% that year), and that
  flattering early period is exactly why it got published.

## Running it

The bot runs once near the close as a k8s CronJob
(`30 12,15 * * 1-5` America/New_York). Each run:

1. Checks the Alpaca clock and calendar. It only acts when it's within
   `RUN_WINDOW_MINUTES` of the close and before the 15:50 MOC cutoff. That
   handles half days.
2. Builds history from Alpaca adjusted daily bars and uses today's latest
   IEX trade as a stand-in for the close.
3. Submits whole-share **market-on-close** orders, sells first.
4. Appends one JSON line per run to `/app/logs/runs.jsonl`.

There is no state file. Positions are reconciled against the targets on every
run, so a missed day fixes itself the next day. A retry is safe because a run
cancels its own open orders before resubmitting.

**It needs a dedicated Alpaca paper account.** The bot sizes off the whole
account's equity. It refuses to trade if it sees positions outside its
universe, so it can't share an account with lowfloat_trader or tradingbot.

### Deploy (Flux, same pattern as lowfloat_trader)

1. Create a new Alpaca paper account and copy `k8s/secret.env.example` to
   `k8s/secret.env` with its keys. Then run `sops -e -i k8s/secret.env`.
2. Push this repo to GitHub. The workflow runs the tests and pushes
   `ghcr.io/parthjp80/etf_trader:latest`. Make the package public, or add a
   pull secret.
3. Add a `GitRepository` and `Kustomization` for `./k8s` in `flux-system`,
   with a deploy key and sops decryption, like lowfloat_trader.
4. To test without placing orders, set `DRY_RUN=true` and trigger a run:
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

`data/` holds the cached Yahoo bars so the backtest can be reproduced. Pass
`--refresh` to re-download them.
