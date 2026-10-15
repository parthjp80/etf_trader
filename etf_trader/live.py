"""
Live runner — one shot, run by a CronJob ~20 minutes before the close.

  1. Bail unless the market is open and we're inside the MOC window.
  2. Build daily closes from Alpaca history, with today's latest trade
     standing in for today's close.
  3. Compute target weights with strategy.target_weights() — the same
     function the backtest grades.
  4. Diff against current positions and submit market-on-close orders.

A rotation sells SPY and buys SSO (or back) in the same closing auction, so
the buy is submitted before the sell's proceeds exist. That needs a margin
account (Alpaca paper accounts are margin by default); the bot refuses to
run on a cash account.

The strategy is a pure function of price history, so there's no state file:
a missed day is reconciled on the next run (anything held with a zero
target is sold, anything targeted but not held is bought).

The bot assumes it owns the whole account. It refuses to trade if it finds
positions outside its universe, so it can't fight another bot.
"""

import json
import logging
import math
import os
import sys
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from . import strategy

ET = ZoneInfo("America/New_York")
PAPER_URL = "https://paper-api.alpaca.markets"
DATA_URL = "https://data.alpaca.markets"

# Alpaca rejects market-on-close orders after 15:50 ET (10 min before close)
MOC_CUTOFF_MINUTES = 10
# Size off slightly less than full equity: the close can differ from the
# price we size at ~30 min earlier
CASH_BUFFER = 0.01

log = logging.getLogger("etf_trader")


class Config:
    def __init__(self, env=os.environ):
        self.key_id = env["APCA_KEY_ID"]
        self.secret = env["APCA_SECRET_KEY"]
        self.base_url = env.get("APCA_BASE_URL", PAPER_URL).rstrip("/")
        self.allow_live = env.get("ALLOW_LIVE", "false").lower() == "true"
        self.dry_run = env.get("DRY_RUN", "false").lower() == "true"
        self.window_minutes = int(env.get("RUN_WINDOW_MINUTES", "40"))
        self.log_path = Path(env.get("LOG_PATH", "/app/logs"))
        if self.base_url != PAPER_URL and not self.allow_live:
            raise SystemExit(f"Refusing to trade against {self.base_url} without ALLOW_LIVE=true")


class Alpaca:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.s = requests.Session()
        self.s.headers.update({
            "APCA-API-KEY-ID": cfg.key_id,
            "APCA-API-SECRET-KEY": cfg.secret,
        })

    def _req(self, method, url, **kw):
        r = self.s.request(method, url, timeout=20, **kw)
        if not r.ok:
            raise RuntimeError(f"{method} {url} → {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else None

    def get(self, path, **params):
        return self._req("GET", f"{self.cfg.base_url}{path}", params=params)

    def data(self, path, **params):
        return self._req("GET", f"{DATA_URL}{path}", params=params)

    def daily_closes(self, symbols, start: date, end: date) -> pd.DataFrame:
        """Split/dividend-adjusted daily closes, start..end inclusive."""
        bars, token = {}, None
        while True:
            params = dict(
                symbols=",".join(symbols), timeframe="1Day",
                start=start.isoformat(), end=end.isoformat(),
                adjustment="all", feed="sip", limit=10000,
            )
            if token:
                params["page_token"] = token
            page = self.data("/v2/stocks/bars", **params)
            for sym, rows in (page.get("bars") or {}).items():
                bars.setdefault(sym, []).extend(rows)
            token = page.get("next_page_token")
            if not token:
                break
        frame = {
            sym: pd.Series({pd.Timestamp(b["t"]).tz_convert(ET).date(): b["c"] for b in rows})
            for sym, rows in bars.items()
        }
        df = pd.DataFrame(frame)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def latest_prices(self, symbols) -> dict[str, float]:
        # Free plan: real-time quotes are IEX only. Fine for these ETFs,
        # which trade a penny wide.
        trades = self.data("/v2/stocks/trades/latest", symbols=",".join(symbols), feed="iex")["trades"]
        return {sym: float(t["p"]) for sym, t in trades.items()}

    def cancel_order(self, order_id):
        self._req("DELETE", f"{self.cfg.base_url}/v2/orders/{order_id}")

    def submit(self, symbol, qty, side, client_order_id):
        return self._req("POST", f"{self.cfg.base_url}/v2/orders", json={
            "symbol": symbol, "qty": str(qty), "side": side,
            "type": "market", "time_in_force": "cls",
            "client_order_id": client_order_id,
        })


def plan_orders(
    w_today: dict[str, float],
    w_yesterday: dict[str, float],
    positions: dict[str, int],
    equity: float,
    prices: dict[str, float],
) -> list[tuple[str, str, int]]:
    """
    Return [(symbol, side, qty)], sells first.

    Mirrors backtest.run(): a symbol is re-sized only when its target weight
    changed. On top of that, positions that disagree with the target in kind
    (held but targeted at 0, or targeted but not held) are always fixed —
    that's how a missed run self-heals.
    """
    orders = []
    for sym in strategy.universe():
        target_w = w_today.get(sym, 0.0)
        held = positions.get(sym, 0)
        budget = target_w * equity * (1 - CASH_BUFFER)
        target_qty = math.floor(budget / prices[sym]) if target_w > 0 else 0
        should_trade = (
            target_w != w_yesterday.get(sym, 0.0)
            or (target_w == 0 and held != 0)
            or (target_w > 0 and held == 0)
        )
        delta = target_qty - held
        if should_trade and delta:
            orders.append((sym, "buy" if delta > 0 else "sell", abs(delta)))
    return sorted(orders, key=lambda o: o[1] != "sell")


def run(cfg: Config, now: datetime | None = None) -> dict:
    api = Alpaca(cfg)
    now = now or datetime.now(ET)
    today = now.date()
    record = {"ts": now.isoformat(), "dry_run": cfg.dry_run}

    clock = api.get("/v2/clock")
    if not clock["is_open"]:
        return {**record, "action": "skip", "reason": "market closed"}

    calendar = api.get("/v2/calendar", start=today.isoformat(), end=today.isoformat())
    today_cal = next(d for d in calendar if d["date"] == today.isoformat())
    close_at = datetime.combine(today, datetime.strptime(today_cal["close"], "%H:%M").time(), ET)
    minutes_left = (close_at - now).total_seconds() / 60
    if not (MOC_CUTOFF_MINUTES < minutes_left <= cfg.window_minutes):
        return {**record, "action": "skip", "reason": f"{minutes_left:.0f} min to close, outside window"}

    symbols = strategy.universe()
    core = strategy.CORE_SYMBOL
    # ~400 trading days: SMA(200) warm-up plus plenty of room for the dip
    # rule's state to settle
    closes = api.daily_closes([core], today - timedelta(days=600), today - timedelta(days=1))
    spy = closes.loc[closes.index.date < today, core].dropna()
    if len(spy) < strategy.REGIME_SMA_DAYS + 50:
        raise RuntimeError(f"only {len(spy)} days of {core} history")
    prices = api.latest_prices(symbols)
    spy.loc[pd.Timestamp(today)] = prices[core]

    weights = strategy.target_weights(spy)
    w_today = weights.iloc[-1].to_dict()
    w_yesterday = weights.iloc[-2].to_dict()

    account = api.get("/v2/account")
    if int(account.get("multiplier", "1")) < 2:
        raise RuntimeError("cash account: same-close rotations need a margin account")
    equity = float(account["equity"])
    positions = {p["symbol"]: int(float(p["qty"])) for p in api.get("/v2/positions")}
    foreign = sorted(set(positions) - set(symbols))
    if foreign:
        raise RuntimeError(f"account holds positions outside the strategy: {foreign} — "
                           "this bot needs a dedicated account")

    # Idempotent re-runs: drop any MOC orders a previous attempt left today
    open_orders = api.get("/v2/orders", status="open", symbols=",".join(symbols))
    if not cfg.dry_run:
        for o in open_orders:
            api.cancel_order(o["id"])

    orders = plan_orders(w_today, w_yesterday, positions, equity, prices)
    submitted = []
    for sym, side, qty in orders:
        coid = f"etf-{today:%Y%m%d}-{sym}-{side}-{uuid.uuid4().hex[:8]}"
        if not cfg.dry_run:
            api.submit(sym, qty, side, coid)
        submitted.append({"symbol": sym, "side": side, "qty": qty, "ref_price": prices[sym]})

    return {
        **record,
        "action": "trade" if submitted else "hold",
        "equity": equity,
        "spy_rsi2": round(float(strategy.rsi(spy).iloc[-1]), 1),
        "boosted": bool(w_today[strategy.BOOST_SYMBOL]),
        "targets": {k: v for k, v in w_today.items() if v},
        "positions": positions,
        "cancelled_open_orders": len(open_orders),
        "orders": submitted,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config()
    try:
        record = run(cfg)
    except Exception as e:
        record = {"ts": datetime.now(ET).isoformat(), "action": "error", "error": str(e)}
        log.exception("run failed")
    log.info(json.dumps(record, default=str))
    cfg.log_path.mkdir(parents=True, exist_ok=True)
    with open(cfg.log_path / "runs.jsonl", "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    sys.exit(1 if record["action"] == "error" else 0)


if __name__ == "__main__":
    main()
