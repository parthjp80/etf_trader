import numpy as np
import pandas as pd
import pytest

from etf_trader import backtest, live, strategy


def make_closes(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    data = {
        sym: 100 * np.exp(np.cumsum(rng.normal(0.0003 * (i + 1), 0.012, n)))
        for i, sym in enumerate(strategy.universe())
    }
    return pd.DataFrame(data, index=idx)


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 30, dtype=float))
    down = up[::-1].reset_index(drop=True)
    assert strategy.rsi(up).iloc[-1] == pytest.approx(100)
    assert strategy.rsi(down).iloc[-1] == pytest.approx(0)
    r = strategy.rsi(make_closes()["SPY"]).dropna()
    assert r.between(0, 100).all()


def test_mean_reversion_enters_on_dip_and_exits_above_sma5():
    # Long steady uptrend, then a two-day dip, then recovery
    base = list(np.linspace(100, 160, 220))
    prices = base + [155, 150, 151, 158, 162]
    close = pd.Series(prices, index=pd.bdate_range("2021-01-01", periods=len(prices)))
    pos = strategy.mean_reversion_position(close)
    assert not pos.iloc[219]
    assert pos.iloc[221]           # RSI(2) collapses on the dip, still above SMA200
    assert not pos.iloc[-1]        # recovered above SMA5 → flat


def test_month_end_flags():
    idx = pd.bdate_range("2024-01-25", "2024-03-05")
    flags = strategy.month_end_flags(idx, last_is_month_end=False)
    assert list(idx[flags]) == [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")]
    assert strategy.month_end_flags(idx, last_is_month_end=True).iloc[-1]


def test_weights_are_long_only_and_never_levered():
    closes = make_closes()
    me = strategy.month_end_flags(closes.index, False)
    w = strategy.target_weights(closes, me)
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1 + 1e-12).all()
    # Trend sleeve fully invested after warm-up
    assert w.iloc[-1].sum() >= strategy.TREND_WEIGHT


def test_no_lookahead():
    """Changing future prices must not change any past weight."""
    closes = make_closes()
    me = strategy.month_end_flags(closes.index, False)
    w = strategy.target_weights(closes, me)

    cut = 500
    shocked = closes.copy()
    shocked.iloc[cut:] *= np.linspace(0.5, 2.0, len(closes) - cut)[:, None]
    w2 = strategy.target_weights(shocked, me)
    pd.testing.assert_frame_equal(w.iloc[:cut], w2.iloc[:cut])


def test_backtest_flat_weights_keep_cash():
    closes = make_closes(300)
    me = strategy.month_end_flags(closes.index, False)
    w = pd.DataFrame(0.0, index=closes.index, columns=strategy.universe())
    res = backtest.run(closes, closes, w, me)
    assert res.equity.eq(100_000).all()
    assert res.trades.empty


def test_backtest_buy_and_hold_matches_price_less_costs():
    closes = make_closes(300)
    me = pd.Series(False, index=closes.index)
    w = pd.DataFrame(0.0, index=closes.index, columns=strategy.universe())
    w["SPY"] = 1.0
    res = backtest.run(closes, closes, w, me, cost_bps=10)
    expected = closes["SPY"].iloc[-1] / closes["SPY"].iloc[0] * 100_000 * (1 - 0.001)
    assert res.equity.iloc[-1] == pytest.approx(expected, rel=1e-3)
    assert len(res.trades) == 1


def test_plan_orders_sells_first_and_heals_missed_runs():
    prices = {s: 100.0 for s in strategy.universe()}
    w = {"SPY": 0.25, "GLD": 0.5}
    orders = live.plan_orders(
        w_today=w, w_yesterday=w, month_end=False,
        positions={"TLT": 500, "SPY": 250}, equity=100_000, prices=prices,
    )
    # TLT held but untargeted → sold; GLD targeted but missing → bought;
    # SPY unchanged target and already held → left alone
    assert orders == [("TLT", "sell", 500), ("GLD", "buy", 500)]


def test_plan_orders_resizes_only_on_change_or_month_end():
    prices = {s: 100.0 for s in strategy.universe()}
    w = {"GLD": 0.5}
    drifted = {"GLD": 450}
    assert live.plan_orders(w, w, False, drifted, 100_000, prices) == []
    assert live.plan_orders(w, w, True, drifted, 100_000, prices) == [("GLD", "buy", 50)]


def test_is_month_end():
    cal = [{"date": "2026-10-29"}, {"date": "2026-10-30"}, {"date": "2026-11-02"}]
    assert live.is_month_end(cal, pd.Timestamp("2026-10-30").date())
    assert not live.is_month_end(cal, pd.Timestamp("2026-10-29").date())


def test_refuses_live_endpoint_without_flag():
    env = {"APCA_KEY_ID": "k", "APCA_SECRET_KEY": "s", "APCA_BASE_URL": "https://api.alpaca.markets"}
    with pytest.raises(SystemExit):
        live.Config(env)
