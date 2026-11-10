"""
Claude as a second look before a sleeve boosts into 2x.

Where it sits in a live run (see live.run):

    closes -> strategy.target_weights()
        -> ai_review.Reviewer.review()   <- here, only on a day a boost starts
        -> ai_review.apply()             (enforce mode only)
        -> plan_orders() -> Alpaca

When a sleeve's dip rule fires, Claude gets the numbers behind the signal,
recent SPY/QQQ headlines, and the account, and answers APPROVE (boost as
the rule says), REDUCE (boost only part of the sleeve) or VETO (stay in
the core ETF). Its power runs one way only. It cannot start a boost the
rule didn't fire, add leverage, delay an exit, or touch the core sleeves.
The rule's exit still ends every window.

The decision covers the whole boost window, from the day the rule fires
until it exits, so a veto isn't undone the next day. Decisions are kept in
<LOG_PATH>/ai_reviews.jsonl, keyed by (core, window start). That file is
also how report.py grades the reviewer: every window's 2x-minus-1x edge is
known from prices whether or not the veto was enforced, so in shadow mode
the vetoed windows still show what they would have cost or saved.

Modes (AI_REVIEW): off | shadow (ask and log, trade as if off) | enforce.
"""

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import pandas as pd

from . import strategy

log = logging.getLogger("etf_trader.ai_review")

MODES = ("off", "shadow", "enforce")
FAIL_MODES = ("open", "closed")
DEFAULT_MODEL = "claude-opus-5-5"
ACTIONS = ("approve", "reduce", "veto")
MIN_REDUCE, MAX_REDUCE = 0.25, 0.75
RISK_FLAGS = ["macro_event", "systemic_shock", "news_catalyst", "data_quality", "other"]

SYSTEM_PROMPT = f"""You review one decision for an automated ETF bot trading a small account: whether to let a sleeve rotate into its 2x ETF today.

The strategy, fixed and not yours to change:
- The account is split 50/50 into a SPY sleeve and a QQQ sleeve. Each sleeve normally holds its index ETF.
- Dip rule (Connors & Alvarez, 2008), on each index's own adjusted daily closes: when close > SMA(200) and RSI(2) < {strategy.RSI_ENTRY:g}, the sleeve rotates into its 2x ETF (SSO for SPY, QLD for QQQ). It rotates back at the first close above SMA(5), usually a few trading days later.
- Measured since 2009 (out of sample for the rule): about 150 boost windows per sleeve, average edge of the 2x over the 1x about +0.29% per window after costs, t-stat about 2. A real but thin edge, made of many small wins and some sharp losses.
- Orders go in about 15 minutes before today's close, so the boost would be held from today's close.

You see a sleeve only on the day its rule fires. You answer for the whole window, until the rule exits.

What you can do:
- APPROVE (the default): boost as the rule says.
- REDUCE (size_multiplier {MIN_REDUCE}-{MAX_REDUCE}): boost only that fraction of the sleeve; the rest stays in the 1x ETF.
- VETO: the sleeve stays in its 1x ETF for this window.
You cannot add leverage, start a boost the rule didn't fire, or delay the exit.

The rule buys sharp short-term drops. Those days almost always come with fearful headlines, and that fear is where the edge comes from. "Markets are nervous", "stocks fell on X", a bad week, or a high VIX are not reasons to veto: they describe every entry the rule has ever made.

Veto or reduce only for a concrete, nameable problem the price rule cannot see:
- A scheduled binary macro event (FOMC decision, CPI, payrolls) inside the next few sessions that the headlines make evident. Don't invent dates you aren't sure of.
- Evidence of a systemic shock rather than a pullback: a major bank or broker failing, exchange or market-structure breakdown, a war or attack moving all markets, emergency policy action.
- Prices that look broken (the move in the numbers doesn't match the headlines, or a close looks like a bad print).

Give each reason in one or two sentences, citing the specific headline, date or number it rests on. Review every sleeve you are given exactly once."""


def _schema(cores: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "reviews": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "core": {"type": "string", "enum": cores},
                        "action": {"type": "string", "enum": list(ACTIONS)},
                        "size_multiplier": {
                            "type": "number",
                            "description": f"1 for approve, {MIN_REDUCE}-{MAX_REDUCE} for reduce, 0 for veto.",
                        },
                        "reason": {"type": "string"},
                        "risk_flags": {"type": "array", "items": {"type": "string", "enum": RISK_FLAGS}},
                    },
                    "required": ["core", "action", "size_multiplier", "reason", "risk_flags"],
                    "additionalProperties": False,
                },
            },
            "note": {"type": "string", "description": "One line on the tape overall."},
        },
        "required": ["reviews", "note"],
        "additionalProperties": False,
    }


@dataclass(frozen=True)
class ReviewConfig:
    mode: str = "off"
    model: str = DEFAULT_MODEL
    # What enforce does when Claude can't be reached: "closed" stays in the
    # 1x ETF for that window, "open" boosts as if the reviewer were off.
    # Shadow always fails open.
    fail_mode: str = "closed"
    # The run has ~13 minutes before the close; one retry at most
    timeout_seconds: float = 120.0
    max_headlines: int = 40

    @classmethod
    def from_env(cls, env) -> "ReviewConfig":
        cfg = cls(
            mode=env.get("AI_REVIEW", "off").lower(),
            model=env.get("AI_REVIEW_MODEL", DEFAULT_MODEL),
            fail_mode=env.get("AI_REVIEW_FAIL_MODE", "closed").lower(),
        )
        if cfg.mode not in MODES:
            raise SystemExit(f"AI_REVIEW must be one of {'|'.join(MODES)}, got {cfg.mode!r}")
        if cfg.fail_mode not in FAIL_MODES:
            raise SystemExit(f"AI_REVIEW_FAIL_MODE must be open|closed, got {cfg.fail_mode!r}")
        return cfg


@dataclass
class Decision:
    core: str
    window_start: str        # ISO date the dip rule fired
    action: str              # approve | reduce | veto
    size_multiplier: float   # share of the sleeve held in the 2x ETF
    reason: str
    risk_flags: list = field(default_factory=list)
    source: str = "claude"   # claude | fail_open | fail_closed
    mode: str = "shadow"
    model: str = ""
    review_id: str = ""
    decided_at: str = ""


# ── Boost windows ────────────────────────────────────────────────────────

def window_starts(close: pd.Series) -> pd.Series:
    """For each date, the date its boost window started, or NaT if not boosted."""
    on = strategy.dip_position(close)
    starts = pd.Series(close.index, index=close.index).where(on & ~on.shift(1, fill_value=False))
    return starts.ffill().where(on)


def apply(weights: pd.DataFrame, core_closes: pd.DataFrame, decisions: dict) -> pd.DataFrame:
    """
    Target weights with each reviewed window's boost scaled by its decision.

    Applied to every date of the window, not just today, so plan_orders()
    sees no weight change on day 2 of a vetoed window and leaves it alone.
    """
    out = weights.copy()
    for s in strategy.SLEEVES:
        starts = window_starts(core_closes[s.core])
        for (core, start), d in decisions.items():
            if core != s.core or d.size_multiplier >= 1:
                continue
            days = starts == pd.Timestamp(start)
            out.loc[days, s.boost] = s.weight * d.size_multiplier
            out.loc[days, s.core] = s.weight * (1 - d.size_multiplier)
    return out


def window_edges(core: pd.Series, boost: pd.Series) -> dict[str, dict]:
    """{window start: {exit, edge}} for each finished window: 2x return − 1x return, close to close."""
    starts = window_starts(core)
    out = {}
    for start in map(pd.Timestamp, starts.dropna().unique()):
        held = starts.index[starts == start]
        after = core.index[core.index > held[-1]]
        if not len(after):
            continue  # still open
        exit_ = after[0]
        if pd.isna(boost.get(start)) or pd.isna(boost.get(exit_)):
            continue
        edge = (boost[exit_] / boost[start] - 1) - (core[exit_] / core[start] - 1)
        out[start.date().isoformat()] ={"exit": exit_.date().isoformat(), "edge": float(edge)}
    return out


# ── Decision log ─────────────────────────────────────────────────────────

def load_decisions(path: Path) -> dict[tuple[str, str], Decision]:
    """Latest decision per (core, window start)."""
    out = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") != "review":
            continue
        fields = {k: r[k] for k in Decision.__dataclass_fields__ if k in r}
        try:
            d = Decision(**fields)
        except TypeError:
            continue
        out[(d.core, d.window_start)] = d
    return out


# ── Reviewer ─────────────────────────────────────────────────────────────

class Reviewer:
    def __init__(self, cfg: ReviewConfig, log_path: Path, news_fn: Callable | None = None,
                 client=None, now_fn: Callable[[], datetime] = datetime.now):
        self.cfg = cfg
        self.path = log_path
        self.news_fn = news_fn
        self._client = client
        self.now = now_fn

    @property
    def enforcing(self) -> bool:
        return self.cfg.mode == "enforce"

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(timeout=self.cfg.timeout_seconds, max_retries=1)
        return self._client

    def review(self, core_closes: pd.DataFrame, account: dict) -> tuple[dict, list[Decision]]:
        """
        Ask Claude about every sleeve whose boost window starts on the last
        date of core_closes and has no decision yet. Never raises.

        Returns (all decisions on file, the ones made in this call).
        """
        decisions = load_decisions(self.path)
        today = core_closes.index[-1]
        new = []
        for s in strategy.SLEEVES:
            start = window_starts(core_closes[s.core]).iloc[-1]
            if start == today and (s.core, today.date().isoformat()) not in decisions:
                new.append(s)
        if not new:
            return decisions, []

        try:
            fresh = self._ask(core_closes, new, account)
        except Exception as e:
            log.error("AI review failed (%s: %s); fail mode %s", type(e).__name__, e,
                      self.cfg.fail_mode if self.enforcing else "open (shadow)")
            fresh = {}

        made = []
        for s in new:
            d = fresh.get(s.core) or self._fallback(s.core, today, "no decision returned")
            decisions[(d.core, d.window_start)] = d
            made.append(d)
            self._log({"event": "review", **asdict(d)})
            log.info("AI review (%s) %s→%s: %s x%.2f (%s) %s", self.cfg.mode, s.core, s.boost,
                     d.action.upper(), d.size_multiplier, d.source, d.reason)
        return decisions, made

    # ── Claude ──

    def _ask(self, core_closes: pd.DataFrame, sleeves: list, account: dict) -> dict[str, Decision]:
        today = core_closes.index[-1]
        payload = {
            "now_et": self.now().strftime("%Y-%m-%d %H:%M %Z").strip(),
            "boosts_to_review": [s.core for s in sleeves],
            "sleeves": {s.core: self._snapshot(core_closes[s.core], s) for s in strategy.SLEEVES},
            "account": account,
            "headlines": self._headlines(),
        }
        resp = self.client.beta.messages.create(
            model=self.cfg.model,
            max_tokens=16000,
            # On a safety decline, re-run the request on a model that can answer
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            output_config={"effort": "medium",
                           "format": {"type": "json_schema", "schema": _schema(strategy.cores())}},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Review today's boost signal(s):\n\n"
                       + json.dumps(payload, indent=1, default=str)}],
        )
        usage = getattr(resp, "usage", None)
        self._log({"event": "call", "model": self.cfg.model, "stop_reason": resp.stop_reason,
                   "input_tokens": getattr(usage, "input_tokens", None),
                   "output_tokens": getattr(usage, "output_tokens", None)})
        if resp.stop_reason == "refusal":
            raise RuntimeError("model declined the request")
        text = next((b.text for b in resp.content if getattr(b, "type", "") == "text"), None)
        if text is None:
            raise ValueError(f"no text in response (stop_reason {resp.stop_reason})")
        out = json.loads(text)
        if out.get("note"):
            self._log({"event": "note", "note": out["note"]})
        return self._parse(out.get("reviews", []), {s.core for s in sleeves}, today)

    def _parse(self, reviews: list, wanted: set, today: pd.Timestamp) -> dict[str, Decision]:
        out = {}
        for r in reviews:
            core, action = r.get("core"), r.get("action")
            if core not in wanted or core in out or action not in ACTIONS:
                continue
            mult = {"approve": 1.0, "veto": 0.0}.get(action)
            if mult is None:
                mult = min(max(float(r.get("size_multiplier", 0.5)), MIN_REDUCE), MAX_REDUCE)
            flags = [f for f in r.get("risk_flags", []) if f in RISK_FLAGS]
            out[core] = self._decision(core, today, action, mult, str(r.get("reason", "")).strip(),
                                       flags, "claude")
        return out

    def _snapshot(self, c: pd.Series, s) -> dict:
        sma200 = c.rolling(strategy.REGIME_SMA_DAYS).mean().iloc[-1]
        sma5 = c.rolling(strategy.EXIT_SMA_DAYS).mean().iloc[-1]
        last = c.iloc[-11:]
        return {
            "core": s.core,
            "boost_etf": s.boost,
            "rule_fires_today": bool(window_starts(c).iloc[-1] == c.index[-1]),
            "already_boosted": bool(strategy.dip_position(c).iloc[-1]
                                    and window_starts(c).iloc[-1] != c.index[-1]),
            "close_today_approx": round(float(c.iloc[-1]), 2),
            "rsi2": round(float(strategy.rsi(c).iloc[-1]), 1),
            "pct_above_sma200": round(float(c.iloc[-1] / sma200 - 1) * 100, 2),
            "pct_vs_sma5": round(float(c.iloc[-1] / sma5 - 1) * 100, 2),
            "pct_off_52w_high": round(float(c.iloc[-1] / c.iloc[-252:].max() - 1) * 100, 2),
            "daily_returns_pct_last_10": {
                d.date().isoformat(): round(float(r) * 100, 2)
                for d, r in last.pct_change().dropna().items()
            },
        }

    def _headlines(self) -> list:
        if self.news_fn is None:
            return []
        try:
            return self.news_fn(self.cfg.max_headlines)
        except Exception as e:
            log.warning("headline fetch failed: %s", e)
            return []

    # ── helpers ──

    def _decision(self, core, day, action, mult, reason, flags, source) -> Decision:
        return Decision(core=core, window_start=pd.Timestamp(day).date().isoformat(), action=action,
                        size_multiplier=mult, reason=reason, risk_flags=flags, source=source,
                        mode=self.cfg.mode, model=self.cfg.model, review_id=uuid.uuid4().hex[:12],
                        decided_at=self.now().isoformat(timespec="seconds"))

    def _fallback(self, core, day, why) -> Decision:
        if self.enforcing and self.cfg.fail_mode == "closed":
            return self._decision(core, day, "veto", 0.0, f"fail-closed: {why}", [], "fail_closed")
        return self._decision(core, day, "approve", 1.0, f"fail-open: {why}", [], "fail_open")

    def _log(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(json.dumps({"ts": self.now().isoformat(timespec="seconds"), **record}, default=str) + "\n")
