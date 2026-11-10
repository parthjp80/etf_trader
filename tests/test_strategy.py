import numpy as np
import pandas as pd
import pytest

from etf_trader import backtest, history, live, strategy


def make_spy(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))), index=idx)


def make_cores(n=700):
    return pd.DataFrame({"SPY": make_spy(n, seed=0), "QQQ": make_spy(n, seed=1) * 3})


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 30, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert strategy.rsi(up).iloc[-1] == pytest.approx(100)
    assert strategy.rsi(down).iloc[-1] == pytest.approx(0)
    assert strategy.rsi(make_spy()).dropna().between(0, 100).all()


def test_dip_enters_on_dip_and_exits_above_sma5():
    # Long steady uptrend, a two-day dip, then a recovery
    prices = list(np.linspace(100, 160, 220)) + [155, 150, 151, 158, 162]
    close = pd.Series(prices, index=pd.bdate_range("2021-01-01", periods=len(prices)))
    pos = strategy.dip_position(close)
    assert not pos.iloc[219]
    assert pos.iloc[221]           # RSI(2) collapses on the dip, still above SMA200
    assert not pos.iloc[-1]        # recovered above SMA5 → flat


def test_no_dip_buys_below_sma200():
    prices = list(np.linspace(160, 100, 220)) + [95, 90, 88]
    close = pd.Series(prices, index=pd.bdate_range("2021-01-01", periods=len(prices)))
    assert not strategy.dip_position(close).any()


def test_weights_two_independent_half_sleeves():
    cores = make_cores()
    w = strategy.target_weights(cores)
    assert list(w.columns) == ["SPY", "SSO", "QQQ", "QLD"]
    assert ((w["SPY"] + w["SSO"]) == 0.5).all() and ((w["QQQ"] + w["QLD"]) == 0.5).all()
    assert w.isin([0.0, 0.5]).all().all()
    assert w["SSO"].any() and w["QLD"].any()
    # Each sleeve follows only its own index
    pd.testing.assert_series_equal(w["QLD"] > 0, strategy.dip_position(cores["QQQ"]), check_names=False)
    assert ((w["SSO"] > 0) != (w["QLD"] > 0)).any()


def test_no_lookahead():
    """Changing future prices must not change any past weight."""
    spy = make_cores()
    cut = 500
    shocked = spy.copy()
    shocked.iloc[cut:] = shocked.iloc[cut:].mul(np.linspace(0.5, 2.0, len(spy) - cut), axis=0)
    pd.testing.assert_frame_equal(strategy.target_weights(spy).iloc[:cut],
                                  strategy.target_weights(shocked).iloc[:cut])


def _closes(n=300):
    spy = make_spy(n)
    return pd.DataFrame({"SPY": spy, "SSO": (1 + 2 * spy.pct_change().fillna(0)).cumprod() * 100})


def test_backtest_buy_and_hold_matches_price_less_costs():
    closes = _closes()
    w = pd.DataFrame({"SPY": 1.0, "SSO": 0.0}, index=closes.index)
    res = backtest.run(closes, closes, w, cost_bps=10)
    expected = closes["SPY"].iloc[-1] / closes["SPY"].iloc[0] * 100_000 * (1 - 0.001)
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-3)
    assert len(res.trades) == 1


def test_backtest_rotation_trades_both_legs_only_on_change():
    closes = _closes()
    w = pd.DataFrame({"SPY": 1.0, "SSO": 0.0}, index=closes.index)
    w.iloc[100:105] = [0.0, 1.0]
    res = backtest.run(closes, closes, w, cost_bps=0)
    # initial buy, rotate in (2 fills), rotate out (2 fills)
    assert len(res.trades) == 5
    assert set(res.trades.date) == {closes.index[0], closes.index[100], closes.index[105]}


def test_backtest_rebalances_drifted_sleeves_past_tolerance():
    idx = pd.bdate_range("2020-01-01", periods=60)
    up = pd.Series(np.linspace(100, 160, 60), index=idx)   # +60%
    flat = pd.Series(100.0, index=idx)
    closes = pd.DataFrame({"A": up, "B": flat})
    w = pd.DataFrame({"A": 0.5, "B": 0.5}, index=idx)
    banded = backtest.run(closes, closes, w, cost_bps=0)
    drift = backtest.run(closes, closes, w, cost_bps=0, rebalance_tolerance=None)
    assert len(drift.trades) == 2
    assert len(banded.trades) > 2
    # Holdings stay within ~10% of target, so A ends near half the book
    value_a = (banded.trades.groupby("symbol").shares.sum()["A"]) * 160
    assert value_a / banded.equity.iloc[-1] == pytest.approx(0.5, abs=0.06)


def test_synthetic_2x_doubles_daily_returns_before_costs():
    spy = make_spy(50)
    syn = history.synthetic_2x(spy, pd.Series(0.0, index=spy.index), 0.0095)
    daily_cost = (history.FINANCING_SPREAD + 0.0095) / 252
    np.testing.assert_allclose(syn.pct_change().iloc[1:], 2 * spy.pct_change().iloc[1:] - daily_cost)


PRICES = {"SPY": 500.0, "SSO": 100.0, "QQQ": 400.0, "QLD": 80.0}


def test_plan_rotation_sells_first():
    orders = live.plan_orders({"SPY": 0.0, "SSO": 1.0}, {"SPY": 1.0, "SSO": 0.0},
                              {"SPY": 198}, 100_000, PRICES)
    assert orders == [("SPY", "sell", 198), ("SSO", "buy", 990)]


def test_plan_holds_steady_and_heals_missed_runs():
    w = {"SPY": 1.0, "SSO": 0.0}
    # Unchanged target, already held (drifted) → no trade
    assert live.plan_orders(w, w, {"SPY": 180}, 100_000, PRICES) == []
    # Missed the exit: still holding SSO though target is SPY → fix both
    assert live.plan_orders(w, w, {"SSO": 990}, 100_000, PRICES) == [
        ("SSO", "sell", 990), ("SPY", "buy", 198)]


def test_plan_tops_up_partial_fill():
    w = {"SPY": 0.0, "SSO": 1.0}
    # Only 608 of 990 SSO filled yesterday → buy the rest
    assert live.plan_orders(w, w, {"SSO": 608}, 100_000, PRICES) == [("SSO", "buy", 382)]
    # Within tolerance → leave it
    assert live.plan_orders(w, w, {"SSO": 950}, 100_000, PRICES) == []


def test_plan_moves_spy_only_account_to_50_50():
    w = {"SPY": 0.5, "SSO": 0.0, "QQQ": 0.5, "QLD": 0.0}
    # The day the 50/50 version goes live: 100% SPY, unchanged weights
    assert live.plan_orders(w, w, {"SPY": 198}, 100_000, PRICES) == [
        ("SPY", "sell", 99), ("QQQ", "buy", 123)]


def test_plan_rotates_one_sleeve_only():
    before = {"SPY": 0.5, "SSO": 0.0, "QQQ": 0.5, "QLD": 0.0}
    after = {"SPY": 0.5, "SSO": 0.0, "QQQ": 0.0, "QLD": 0.5}
    held = {"SPY": 99, "QQQ": 123}
    assert live.plan_orders(after, before, held, 100_000, PRICES) == [
        ("QQQ", "sell", 123), ("QLD", "buy", 618)]


def test_refuses_live_endpoint_without_flag():
    env = {"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s", "APCA_BASE_URL": "https://api.alpaca.markets"}
    with pytest.raises(SystemExit):
        live.Config(env)


class FakeAlpaca:
    """Stands in for the Alpaca REST API in an end-to-end run()."""

    def __init__(self, cores, positions, multiplier="2", open_orders=()):
        self.cores, self.positions, self.multiplier = cores, positions, multiplier
        self.open_orders, self.submitted, self.cancelled = list(open_orders), [], []

    def __call__(self, cfg):
        return self

    def get(self, path, **params):
        return {
            "/v2/clock": {"is_open": True},
            "/v2/calendar": [{"date": params.get("start"), "close": "16:00"}],
            "/v2/account": {"equity": "100000", "multiplier": self.multiplier},
            "/v2/positions": [{"symbol": s, "qty": str(q)} for s, q in self.positions.items()],
            "/v2/orders": self.open_orders,
        }[path]

    def daily_closes(self, symbols, start, end):
        return self.cores.loc[:pd.Timestamp(end), symbols]

    def latest_prices(self, symbols):
        return {**self.cores.iloc[-1].to_dict(), "SSO": 100.0, "QLD": 80.0}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    def submit(self, symbol, qty, side, client_order_id):
        self.submitted.append((symbol, side, qty))


def _run(monkeypatch, fake, minutes_before_close=15, dry_run=False):
    monkeypatch.setattr(live, "Alpaca", fake)
    env = {"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s", "DRY_RUN": str(dry_run).lower()}
    today = fake.cores.index[-1]
    now = (today + pd.Timedelta(hours=16) - pd.Timedelta(minutes=minutes_before_close)).tz_localize(live.ET)
    return live.run(live.Config(env), now=now.to_pydatetime())


def test_run_end_to_end_buys_both_cores_in_empty_account(monkeypatch):
    cores = make_cores(500)
    cores.iloc[-1] = cores.iloc[-2] * 1.01  # up day → no dip signal today
    fake = FakeAlpaca(cores, positions={}, open_orders=[{"id": "old"}])
    rec = _run(monkeypatch, fake)
    assert rec["action"] == "trade"
    assert not any(s["boosted"] for s in rec["sleeves"].values())
    qty = {c: int(50_000 * 0.99 // cores[c].iloc[-1]) for c in ("SPY", "QQQ")}
    assert fake.submitted == [("SPY", "buy", qty["SPY"]), ("QQQ", "buy", qty["QQQ"])]
    assert fake.cancelled == ["old"]


def test_run_skips_outside_window_and_respects_dry_run(monkeypatch):
    spy = make_cores(500)
    assert _run(monkeypatch, FakeAlpaca(spy, {}), minutes_before_close=30)["action"] == "skip"
    assert _run(monkeypatch, FakeAlpaca(spy, {}), minutes_before_close=1)["action"] == "skip"
    fake = FakeAlpaca(spy, {})
    rec = _run(monkeypatch, fake, dry_run=True)
    assert rec["orders"] and not fake.submitted


def test_run_refuses_foreign_positions_and_cash_accounts(monkeypatch):
    spy = make_cores(500)
    with pytest.raises(RuntimeError, match="outside the strategy"):
        _run(monkeypatch, FakeAlpaca(spy, {"AAPL": 5}))
    with pytest.raises(RuntimeError, match="margin"):
        _run(monkeypatch, FakeAlpaca(spy, {}, multiplier="1"))


from etf_trader import report  # noqa: E402


class FakeReportApi:
    def __init__(self, spy):
        self.spy = spy

    def get(self, path, **params):
        return {
            "/v2/account": {"equity": "102000", "last_equity": "101000", "cash": "800"},
            "/v2/positions": [{"symbol": "SSO", "qty": "1000", "avg_entry_price": "100", "current_price": "101.2",
                               "market_value": "101200", "unrealized_pl": "1200", "unrealized_plpc": "0.012"}],
            "/v2/orders": [{"symbol": "SSO", "side": "buy", "qty": "500", "filled_qty": "200",
                            "filled_avg_price": "101.2", "status": "expired"}],
        }[path]

    def daily_closes(self, symbols, start, end):
        return self.spy


def test_report_numbers_and_warnings():
    spy = make_cores(500)
    start, today = spy.index[-10], spy.index[-1]
    runs = [{"ts": f"{start:%Y-%m-%d}T15:30:00-04:00", "action": "trade", "equity": 100000.0, "orders": []},
            {"ts": f"{today:%Y-%m-%d}T15:30:00-04:00", "action": "hold", "orders": []}]
    now = (today + pd.Timedelta(hours=16, minutes=20)).tz_localize(live.ET).to_pydatetime()
    d = report.gather(FakeReportApi(spy), runs, now)
    assert d["day_pnl"] == 1000
    assert d["since"]["bot"] == pytest.approx(0.02)
    rets = spy.iloc[-1] / spy.loc[start] - 1
    assert d["since"]["spy"] == pytest.approx(rets["SPY"])
    assert d["since"]["mix"] == pytest.approx(rets.mean())
    for g in d["signals"]:
        assert g["exit_level"] == pytest.approx(spy[g["core"]].iloc[-4:].mean())
    assert any("200 of 500 filled" in w for w in d["warnings"])
    subject, html = report.render(d)
    assert "▲ $1,000.00" in subject and "⚠️" in subject
    assert "$101,200.00" in html and "Needs a look" in html


def test_report_flags_missing_run():
    spy = make_cores(500)
    now = (spy.index[-1] + pd.Timedelta(hours=16, minutes=20)).tz_localize(live.ET).to_pydatetime()
    d = report.gather(FakeReportApi(spy), [], now)
    assert d["since"] is None
    assert any("No run reached the closing window" in w for w in d["warnings"])
    report.render(d)


def test_submits_plain_market_day_orders(monkeypatch):
    api = live.Alpaca(live.Config({"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s"}))
    sent = {}
    monkeypatch.setattr(api, "_req", lambda method, url, **kw: sent.update(kw["json"]))
    api.submit("SSO", 497, "buy", "etf-x")
    assert sent["type"] == "market" and sent["time_in_force"] == "day"


# ── AI review ──

import json  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from etf_trader import ai_review  # noqa: E402


def dip_today(n=500):
    """Cores in a steady uptrend whose last close drops 3%: both rules fire today."""
    idx = pd.bdate_range("2020-01-01", periods=n)
    up = pd.Series(100 * 1.001 ** np.arange(n), index=idx)
    cores = pd.DataFrame({"SPY": up, "QQQ": up * 3})
    cores.iloc[-1] = cores.iloc[-2] * 0.97
    return cores


class FakeClaude:
    def __init__(self, reviews=None, error=None):
        self.reviews, self.error, self.calls = reviews, error, []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        text = json.dumps({"reviews": self.reviews, "note": "n"})
        return SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text=text)],
                               usage=SimpleNamespace(input_tokens=1, output_tokens=1))


def _reviewer(tmp_path, client, mode="enforce", fail_mode="closed"):
    cfg = ai_review.ReviewConfig(mode=mode, fail_mode=fail_mode)
    return ai_review.Reviewer(cfg, tmp_path / "ai_reviews.jsonl", news_fn=lambda n: [], client=client)


def test_window_starts_marks_each_boost_window():
    cores = dip_today()
    on = strategy.dip_position(cores["SPY"])
    assert on.iloc[-1] and not on.iloc[-2]
    starts = ai_review.window_starts(cores["SPY"])
    assert starts.iloc[-1] == cores.index[-1]
    assert starts[~on].isna().all()


def test_review_asks_once_per_window_and_apply_holds_for_the_window(tmp_path):
    cores = dip_today()
    today = cores.index[-1]
    assert ai_review.window_starts(cores["SPY"]).iloc[-1] == today
    claude = FakeClaude([
        {"core": "SPY", "action": "veto", "size_multiplier": 0, "reason": "FOMC tomorrow", "risk_flags": ["macro_event"]},
        {"core": "QQQ", "action": "reduce", "size_multiplier": 0.9, "reason": "r", "risk_flags": ["bogus"]},
    ])
    rv = _reviewer(tmp_path, claude)
    decisions, made = rv.review(cores, {"equity": 1})
    assert len(claude.calls) == 1 and len(made) == 2
    assert claude.calls[0]["output_config"]["format"]["type"] == "json_schema"
    spy, qqq = decisions[("SPY", today.date().isoformat())], decisions[("QQQ", today.date().isoformat())]
    assert spy.action == "veto" and spy.size_multiplier == 0 and spy.risk_flags == ["macro_event"]
    assert qqq.size_multiplier == ai_review.MAX_REDUCE and qqq.risk_flags == []

    # Same day again (a retry): reuses the logged decisions, no new call
    assert rv.review(cores, {})[1] == [] and len(claude.calls) == 1

    # Next day, still in the window: the veto still applies, and yesterday's
    # weights agree, so plan_orders sees nothing to change
    nxt = pd.concat([cores, pd.DataFrame([cores.iloc[-1] * 0.99], index=[today + pd.offsets.BDay()])])
    assert strategy.dip_position(nxt["SPY"]).iloc[-1]
    decisions, made = rv.review(nxt, {})
    assert made == []
    w = ai_review.apply(strategy.target_weights(nxt), nxt, decisions)
    assert w["SSO"].iloc[-1] == 0 and w["SPY"].iloc[-1] == 0.5
    assert w["SSO"].iloc[-2] == 0 and w["SPY"].iloc[-2] == 0.5
    assert w["QLD"].iloc[-1] == pytest.approx(0.5 * 0.75) and w["QQQ"].iloc[-1] == pytest.approx(0.5 * 0.25)


def test_review_failure_modes(tmp_path):
    cores = dip_today()
    boom = FakeClaude(error=RuntimeError("down"))
    _, made = _reviewer(tmp_path / "a", boom, "enforce", "closed").review(cores, {})
    assert {d.action for d in made} == {"veto"} and {d.source for d in made} == {"fail_closed"}
    _, made = _reviewer(tmp_path / "b", boom, "enforce", "open").review(cores, {})
    assert {d.action for d in made} == {"approve"}
    _, made = _reviewer(tmp_path / "c", boom, "shadow", "closed").review(cores, {})
    assert {d.action for d in made} == {"approve"} and {d.source for d in made} == {"fail_open"}
    # Claude leaving a sleeve out counts as a failure for that sleeve
    partial = FakeClaude([{"core": "SPY", "action": "approve", "size_multiplier": 1, "reason": "", "risk_flags": []}])
    decisions, _ = _reviewer(tmp_path / "d", partial).review(cores, {})
    assert {d.core: d.source for d in decisions.values()} == {"SPY": "claude", "QQQ": "fail_closed"}


def test_review_not_called_without_a_new_boost(tmp_path):
    claude = FakeClaude([])
    cores = make_cores(500)
    cores.iloc[-1] = cores.iloc[-2] * 1.01
    assert _reviewer(tmp_path, claude).review(cores, {}) == ({}, [])
    assert claude.calls == []


def test_review_config_validates():
    assert ai_review.ReviewConfig.from_env({}).mode == "off"
    with pytest.raises(SystemExit):
        ai_review.ReviewConfig.from_env({"AI_REVIEW": "yes"})
    with pytest.raises(SystemExit):
        ai_review.ReviewConfig.from_env({"AI_REVIEW_FAIL_MODE": "maybe"})


def test_run_enforced_veto_keeps_core(monkeypatch, tmp_path):
    cores = dip_today()
    claude = FakeClaude([
        {"core": "SPY", "action": "veto", "size_multiplier": 0, "reason": "x", "risk_flags": []},
        {"core": "QQQ", "action": "approve", "size_multiplier": 1, "reason": "y", "risk_flags": []},
    ])
    real = ai_review.Reviewer.__init__
    monkeypatch.setattr(ai_review.Reviewer, "__init__",
                        lambda self, *a, **kw: real(self, *a, **{**kw, "client": claude}))
    fake = FakeAlpaca(cores, positions={"SPY": 100, "QQQ": 50})
    fake.news = lambda limit: []
    monkeypatch.setattr(live, "Alpaca", fake)
    env = {"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s", "AI_REVIEW": "enforce", "LOG_PATH": str(tmp_path)}
    now = (cores.index[-1] + pd.Timedelta(hours=15, minutes=45)).tz_localize(live.ET).to_pydatetime()
    rec = live.run(live.Config(env), now=now)
    assert rec["ai_review"]["SPY"]["action"] == "veto"
    syms = {s for s, _, _ in fake.submitted}
    assert "SSO" not in syms and "QLD" in syms
    assert ("QQQ", "sell", 50) in fake.submitted

    # Shadow: logged, but both sleeves boost
    fake2 = FakeAlpaca(cores, positions={"SPY": 100, "QQQ": 50})
    fake2.news = lambda limit: []
    monkeypatch.setattr(live, "Alpaca", fake2)
    rec = live.run(live.Config({**env, "AI_REVIEW": "shadow", "LOG_PATH": str(tmp_path / "s")}), now=now)
    assert {s for s, _, _ in fake2.submitted} >= {"SSO", "QLD"}


def test_report_grades_finished_windows():
    cores = dip_today(500)
    # Rally after the dip so the window exits
    later = pd.bdate_range(cores.index[-1] + pd.offsets.BDay(), periods=3)
    tail = pd.DataFrame({"SPY": cores["SPY"].iloc[-1] * 1.02 ** np.arange(1, 4),
                         "QQQ": cores["QQQ"].iloc[-1] * 1.02 ** np.arange(1, 4)}, index=later)
    cl = pd.concat([cores, tail])
    start = ai_review.window_starts(cl["SPY"]).dropna().iloc[-1]
    lev = pd.DataFrame({"SSO": cl["SPY"] ** 2 / 100, "QLD": cl["QQQ"] ** 2 / 300}, index=cl.index)
    edges = ai_review.window_edges(cl["SPY"], lev["SSO"])
    assert edges[start.date().isoformat()]["edge"] > 0

    class Api:
        def daily_closes(self, symbols, start, end):
            return lev
    d = ai_review.Decision(core="SPY", window_start=start.date().isoformat(), action="veto",
                           size_multiplier=0, reason="r", source="claude")
    g = report.grade_reviews(Api(), cl, {("SPY", d.window_start): d}, cl.index[-1])
    assert len(g["graded"]["veto"]) == 1 and g["open"] == {}
