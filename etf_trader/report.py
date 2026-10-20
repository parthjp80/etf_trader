"""
End-of-day P&L email, run by its own CronJob after the close.

    python -m etf_trader.report            # build and send
    python -m etf_trader.report --stdout   # print the HTML instead

Reads the account, positions and today's orders from Alpaca, and the bot's
own run log (runs.jsonl) for today's decision and the starting equity. The
since-start return is compared against SPY held over the same span, which
and against the same 50/50 SPY/QQQ mix held without boosts: the
benchmarks the strategy has to beat.

Email settings come from the environment, same as lowfloat_trader:
EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD (Gmail app password),
EMAIL_SMTP_HOST, EMAIL_SMTP_PORT.
"""

import json
import logging
import os
import smtplib
import sys
from datetime import datetime, time, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from . import live, strategy

ET = live.ET
GREEN, RED, AMBER = "#3B6D11", "#A32D2D", "#8A6D1F"
# Free data plan: SIP bars from the last 15 minutes are off limits
SIP_DELAY = timedelta(minutes=16)

log = logging.getLogger("etf_trader.report")


def load_runs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    runs = []
    for line in path.read_text().splitlines():
        try:
            runs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return runs


def gather(api, runs: list[dict], now: datetime) -> dict:
    today = now.date()
    account = api.get("/v2/account")
    positions = api.get("/v2/positions")
    orders = api.get("/v2/orders", status="all", direction="asc",
                     after=datetime.combine(today, time(), ET).isoformat(),
                     symbols=",".join(strategy.universe()))

    cores = strategy.cores()
    closes = api.daily_closes(cores, today - timedelta(days=600),
                              (now - SIP_DELAY).replace(microsecond=0)).reindex(columns=cores).dropna()
    equity = float(account["equity"])
    last_equity = float(account.get("last_equity") or equity)

    # Each sleeve's state after today's close, and what flips it tomorrow
    signals = []
    for sl in strategy.SLEEVES:
        c = closes[sl.core]
        signals.append({
            "core": sl.core,
            "boost": sl.boost,
            "boosted": bool(strategy.dip_position(c).iloc[-1]),
            "close": float(c.iloc[-1]),
            "close_date": c.index[-1].date(),
            "rsi2": float(strategy.rsi(c).iloc[-1]),
            "sma200": float(c.rolling(strategy.REGIME_SMA_DAYS).mean().iloc[-1]),
            # Tomorrow's close beats tomorrow's SMA(5) iff it beats the mean
            # of the last 4 closes: c > (sum4 + c) / 5  ⇔  c > sum4 / 4
            "exit_level": float(c.iloc[-(strategy.EXIT_SMA_DAYS - 1):].mean()),
        })

    # Since start: from the first run that traded, against SPY and against
    # 50/50 SPY/QQQ bought at that same close
    since = None
    first = next((r for r in runs if r.get("action") == "trade"), None)
    if first:
        start = datetime.fromisoformat(first["ts"]).date()
        from_start = closes.loc[closes.index.date >= start]
        if len(from_start):
            rets = closes.iloc[-1] / from_start.iloc[0] - 1
            since = {
                "date": start,
                "equity": float(first["equity"]),
                "bot": equity / float(first["equity"]) - 1,
                "spy": float(rets["SPY"]),
                "mix": float(sum(sl.weight * rets[sl.core] for sl in strategy.SLEEVES)),
            }

    todays_runs = [r for r in runs if str(r.get("ts", ""))[:10] == today.isoformat()]
    warnings = []
    for r in todays_runs:
        if r.get("action") == "error":
            warnings.append(f"Run at {r['ts'][11:16]} ET failed: {r.get('error', '?')}")
    if not any(r.get("action") in ("trade", "hold") for r in todays_runs):
        warnings.append("No run reached the closing window today, so the bot made no decision.")
    for o in orders:
        filled, qty = float(o.get("filled_qty") or 0), float(o["qty"])
        if o["status"] in ("expired", "canceled") and filled < qty:
            warnings.append(f"{o['side'].title()} {o['symbol']}: {filled:g} of {qty:g} filled, "
                            f"rest {o['status']}. The next run re-sizes it.")

    decision = next((r for r in reversed(todays_runs) if r.get("action") in ("trade", "hold")), None)
    return {
        "now": now,
        "equity": equity,
        "day_pnl": equity - last_equity,
        "day_pct": (equity / last_equity - 1) if last_equity else 0.0,
        "cash": float(account.get("cash", 0)),
        "positions": positions,
        "orders": orders,
        "signals": signals,
        "since": since,
        "decision": decision,
        "warnings": warnings,
    }


def _money(x, sign=True):
    return f"{'+' if sign and x >= 0 else '-' if x < 0 else ''}${abs(x):,.2f}"


def _color(x):
    return GREEN if x >= 0 else RED


def render(d: dict, paper: bool = True) -> tuple[str, str]:
    now, sigs, since = d["now"], d["signals"], d["since"]
    today_str = now.strftime("%A, %B %d %Y")
    arrow = "▲" if d["day_pnl"] >= 0 else "▼"
    td = 'style="padding:8px 10px;color:#555"'
    th = 'style="padding:8px 10px;text-align:left;color:#888;font-weight:500"'

    boosted = [g["boost"] for g in sigs if g["boosted"]]
    mode = " + ".join(f"{g['core']} {'→ ' + g['boost'] + ' (boosted)' if g['boosted'] else 'core'}" for g in sigs)

    def sleeve_line(g):
        head = f"<b>{g['core']}</b> closed ${g['close']:,.2f} ({g['close_date']:%b %d}), RSI(2) {g['rsi2']:.1f}. "
        if g["boosted"]:
            return head + (f"Holding {g['boost']} until {g['core']} closes above <b>${g['exit_level']:,.2f}</b> "
                           f"(its 5-day average), then back to {g['core']} at that close.")
        regime = "above" if g["close"] > g["sma200"] else "below, so no boost possible"
        return head + (f"Holding {g['core']}. Boosts into {g['boost']} when RSI(2) closes under "
                       f"{strategy.RSI_ENTRY:g} while above its 200-day average "
                       f"(${g['sma200']:,.2f}; now {regime}).")

    sleeve_lines = "<br>".join(sleeve_line(g) for g in sigs)

    if d["decision"] and d["decision"].get("orders"):
        placed = ", ".join(f"{o['side']} {o['qty']} {o['symbol']}" for o in d["decision"]["orders"])
        decision = f"Today's run: {placed} at the close."
    elif d["decision"]:
        decision = "Today's run: no change, holding."
    else:
        decision = "Today's run: none."

    cards = [
        (f"${d['equity']:,.0f}", "Equity", "#111"),
        (_money(d["day_pnl"]), f"Today ({d['day_pct']:+.2%})", _color(d["day_pnl"])),
    ]
    if since:
        cards += [
            (f"{since['bot']:+.2%}", f"Bot since {since['date']:%b %d}", _color(since["bot"])),
            (f"{since['mix']:+.2%}", "50/50 held, same span", _color(since["mix"])),
            (f"{since['spy']:+.2%}", "SPY, same span", _color(since["spy"])),
        ]
    card_html = "".join(f'''
    <td style="padding:16px 10px;border-right:1px solid #EBEBEB;text-align:center">
      <div style="font-size:22px;font-weight:700;color:{c}">{v}</div>
      <div style="font-size:12px;color:#888;margin-top:3px">{label}</div>
    </td>''' for v, label, c in cards)

    pos_rows = "".join(f"""
      <tr>
        <td {td}><b style="color:#111">{p['symbol']}</b></td>
        <td {td}>{p['qty']}</td>
        <td {td}>${float(p['avg_entry_price']):,.2f}</td>
        <td {td}>${float(p['current_price']):,.2f}</td>
        <td {td}>${float(p['market_value']):,.2f}</td>
        <td style="padding:8px 10px;font-weight:600;color:{_color(float(p['unrealized_pl']))}">
          {_money(float(p['unrealized_pl']))} ({float(p['unrealized_plpc']):+.2%})</td>
      </tr>""" for p in d["positions"])
    pos_rows += f"""
      <tr><td {td}><b style="color:#111">Cash</b></td><td {td} colspan="3"></td>
        <td {td}>${d['cash']:,.2f}</td><td {td}></td></tr>"""

    def order_row(o):
        filled, qty = float(o.get("filled_qty") or 0), float(o["qty"])
        short = filled < qty and o["status"] != "new"
        price = f"${float(o['filled_avg_price']):,.2f}" if o.get("filled_avg_price") else "—"
        status_color = AMBER if short else "#555"
        return f"""
      <tr>
        <td {td}><b style="color:#111">{o['symbol']}</b></td>
        <td {td}>{o['side'].upper()}</td>
        <td {td}>{qty:g}</td>
        <td {td}>{filled:g}</td>
        <td {td}>{price}</td>
        <td style="padding:8px 10px;color:{status_color}">{o['status'].replace('_', ' ')}</td>
      </tr>"""

    order_rows = "".join(order_row(o) for o in d["orders"]) or \
        '<tr><td colspan="6" style="padding:14px;text-align:center;color:#888">No orders today</td></tr>'

    banner = ""
    if d["warnings"]:
        items = "".join(f'<li style="margin:2px 0">{w}</li>' for w in d["warnings"])
        banner = f"""
  <div style="background:#FFF8E5;border-bottom:1px solid #F0E0B0;padding:12px 24px">
    <div style="font-size:12px;font-weight:600;color:{AMBER};margin-bottom:4px">⚠️ Needs a look</div>
    <ul style="margin:0;padding-left:18px;font-size:12px;color:#7A6320">{items}</ul>
  </div>"""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"></head>
<body style="margin:0;padding:0;background:#F5F5F5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:760px;margin:24px auto;background:#fff;border-radius:10px;overflow:hidden;box-shadow:0 2px 8px rgba(0,0,0,0.08)">

  <table style="width:100%;background:#0F1923;border-collapse:collapse"><tr>
    <td style="padding:22px 26px">
      <div style="color:#fff;font-size:20px;font-weight:600">ETF Trader</div>
      <div style="color:#8A9BB0;font-size:13px;margin-top:2px">End of Day Report — {today_str}</div>
    </td>
    <td style="padding:22px 26px;text-align:right">
      <div style="color:{_color(d['day_pnl'])};font-size:26px;font-weight:700">{arrow} ${abs(d['day_pnl']):,.2f}</div>
      <div style="color:#8A9BB0;font-size:13px">{d['day_pct']:+.2%} account equity change</div>
    </td>
  </tr></table>
{banner}
  <table style="width:100%;border-collapse:collapse;border-bottom:1px solid #EBEBEB"><tr>{card_html}
  </tr></table>

  <div style="padding:18px 24px;border-bottom:1px solid #EBEBEB;font-size:13px;color:#333;line-height:1.6">
    <div style="font-size:14px;font-weight:600;margin-bottom:6px">Signal: {mode}</div>
    {sleeve_lines}<br>
    {decision}
  </div>

  <div style="padding:18px 24px 6px">
    <div style="font-size:14px;font-weight:600;color:#333;margin-bottom:10px">Holdings at the close</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="border-bottom:2px solid #EBEBEB">
        <th {th}>Symbol</th><th {th}>Qty</th><th {th}>Avg cost</th><th {th}>Price</th>
        <th {th}>Value</th><th {th}>Unrealized P&L</th></tr></thead>
      <tbody>{pos_rows}</tbody>
    </table>
  </div>

  <div style="padding:18px 24px">
    <div style="font-size:14px;font-weight:600;color:#333;margin-bottom:10px">Today's orders</div>
    <table style="width:100%;border-collapse:collapse;font-size:13px">
      <thead><tr style="border-bottom:2px solid #EBEBEB">
        <th {th}>Symbol</th><th {th}>Side</th><th {th}>Ordered</th><th {th}>Filled</th>
        <th {th}>Avg fill</th><th {th}>Status</th></tr></thead>
      <tbody>{order_rows}</tbody>
    </table>
  </div>

  <div style="background:#F8F8F8;padding:14px 24px;border-top:1px solid #EBEBEB;font-size:11px;color:#AAA">
    ETF Trader • {'Paper' if paper else 'LIVE'} Trading • Generated {now:%H:%M} ET
  </div>
</div>
</body></html>"""

    subject = (f"📊 ETF Trader — {now:%a %b %d} | {arrow} ${abs(d['day_pnl']):,.2f} "
               f"({d['day_pct']:+.2%}) | {' + '.join(boosted) + ' boost' if boosted else 'no boost'}")
    if d["warnings"]:
        subject += " | ⚠️"
    return subject, html


def send(subject: str, html: str, env=os.environ):
    sender, to, password = env.get("EMAIL_FROM"), env.get("EMAIL_TO"), env.get("EMAIL_PASSWORD")
    if not (sender and to and password):
        raise SystemExit("EMAIL_FROM, EMAIL_TO and EMAIL_PASSWORD must be set")
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, sender, to
    msg.attach(MIMEText(html, "html"))
    with smtplib.SMTP(env.get("EMAIL_SMTP_HOST", "smtp.gmail.com"),
                      int(env.get("EMAIL_SMTP_PORT", "587")), timeout=30) as server:
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, [to], msg.as_string())


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = live.Config()
    api = live.Alpaca(cfg)
    now = datetime.now(ET)
    today = now.date().isoformat()
    if not any(c["date"] == today for c in api.get("/v2/calendar", start=today, end=today)):
        log.info("not a trading day, no report")
        return
    data = gather(api, load_runs(cfg.log_path / "runs.jsonl"), now)
    subject, html = render(data, paper=cfg.base_url == live.PAPER_URL)
    if "--stdout" in sys.argv:
        print(html)
        return
    send(subject, html)
    log.info(f"report sent: {subject}")


if __name__ == "__main__":
    main()
