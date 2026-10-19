import numpy as np
import pandas as pd
import pytest

from etf_trader import backtest, history, live, strategy


def make_spy(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, n))), index=idx)


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


def test_weights_always_fully_invested_in_exactly_one():
    w = strategy.target_weights(make_spy())
    assert (w.sum(axis=1) == 1.0).all()
    assert w.isin([0.0, 1.0]).all().all()
    assert w["SSO"].any() and w["SPY"].any()


def test_no_lookahead():
    """Changing future prices must not change any past weight."""
    spy = make_spy()
    cut = 500
    shocked = spy.copy()
    shocked.iloc[cut:] *= np.linspace(0.5, 2.0, len(spy) - cut)
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


def test_synthetic_2x_doubles_daily_returns_before_costs():
    spy = make_spy(50)
    syn = history.synthetic_2x(spy, pd.Series(0.0, index=spy.index))
    daily_cost = (history.FINANCING_SPREAD + history.SSO_EXPENSE_RATIO) / 252
    np.testing.assert_allclose(syn.pct_change().iloc[1:], 2 * spy.pct_change().iloc[1:] - daily_cost)


PRICES = {"SPY": 500.0, "SSO": 100.0}


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


def test_refuses_live_endpoint_without_flag():
    env = {"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s", "APCA_BASE_URL": "https://api.alpaca.markets"}
    with pytest.raises(SystemExit):
        live.Config(env)


class FakeAlpaca:
    """Stands in for the Alpaca REST API in an end-to-end run()."""

    def __init__(self, spy, positions, multiplier="2", open_orders=()):
        self.spy, self.positions, self.multiplier = spy, positions, multiplier
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
        return self.spy.loc[:pd.Timestamp(end)].to_frame("SPY")

    def latest_prices(self, symbols):
        return {"SPY": float(self.spy.iloc[-1]), "SSO": 100.0}

    def cancel_order(self, order_id):
        self.cancelled.append(order_id)

    def submit(self, symbol, qty, side, client_order_id):
        self.submitted.append((symbol, side, qty))


def _run(monkeypatch, fake, minutes_before_close=25, dry_run=False):
    monkeypatch.setattr(live, "Alpaca", fake)
    env = {"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s", "DRY_RUN": str(dry_run).lower()}
    today = fake.spy.index[-1]
    now = (today + pd.Timedelta(hours=16) - pd.Timedelta(minutes=minutes_before_close)).tz_localize(live.ET)
    return live.run(live.Config(env), now=now.to_pydatetime())


def test_run_end_to_end_buys_spy_in_empty_account(monkeypatch):
    spy = make_spy(500)
    spy.iloc[-1] = spy.iloc[-2] * 1.01  # up day → no dip signal today
    fake = FakeAlpaca(spy, positions={}, open_orders=[{"id": "old"}])
    rec = _run(monkeypatch, fake)
    assert rec["action"] == "trade" and not rec["boosted"]
    qty = int(100_000 * 0.99 // spy.iloc[-1])
    assert fake.submitted == [("SPY", "buy", qty)]
    assert fake.cancelled == ["old"]


def test_run_skips_outside_window_and_respects_dry_run(monkeypatch):
    spy = make_spy(500)
    assert _run(monkeypatch, FakeAlpaca(spy, {}), minutes_before_close=90)["action"] == "skip"
    assert _run(monkeypatch, FakeAlpaca(spy, {}), minutes_before_close=5)["action"] == "skip"
    fake = FakeAlpaca(spy, {})
    rec = _run(monkeypatch, fake, dry_run=True)
    assert rec["orders"] and not fake.submitted


def test_run_refuses_foreign_positions_and_cash_accounts(monkeypatch):
    spy = make_spy(500)
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
        return self.spy.to_frame("SPY")


def test_report_numbers_and_warnings():
    spy = make_spy(500)
    start, today = spy.index[-10], spy.index[-1]
    runs = [{"ts": f"{start:%Y-%m-%d}T15:30:00-04:00", "action": "trade", "equity": 100000.0, "orders": []},
            {"ts": f"{today:%Y-%m-%d}T15:30:00-04:00", "action": "hold", "orders": []}]
    now = (today + pd.Timedelta(hours=16, minutes=20)).tz_localize(live.ET).to_pydatetime()
    d = report.gather(FakeReportApi(spy), runs, now)
    assert d["day_pnl"] == 1000
    assert d["since"]["bot"] == pytest.approx(0.02)
    assert d["since"]["spy"] == pytest.approx(spy.iloc[-1] / spy.loc[start] - 1)
    assert d["signal"]["exit_level"] == pytest.approx(spy.iloc[-4:].mean())
    assert any("200 of 500 filled" in w for w in d["warnings"])
    subject, html = report.render(d)
    assert "▲ $1,000.00" in subject and "⚠️" in subject
    assert "$101,200.00" in html and "Needs a look" in html


def test_report_flags_missing_run():
    spy = make_spy(500)
    now = (spy.index[-1] + pd.Timedelta(hours=16, minutes=20)).tz_localize(live.ET).to_pydatetime()
    d = report.gather(FakeReportApi(spy), [], now)
    assert d["since"] is None
    assert any("No run reached the closing window" in w for w in d["warnings"])
    report.render(d)
