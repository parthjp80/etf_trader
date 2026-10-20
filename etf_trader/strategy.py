"""
Signal logic shared by the backtest and the live runner.

50/50 SPY + QQQ, each with a dip-buy boost:

  Hold two equal sleeves, SPY and QQQ. Each sleeve has its own dip rule,
  computed on its own index. When the rule fires, that sleeve rotates into
  its 2x ETF (SSO for SPY, QLD for QQQ) at the close, and back once the
  rule exits. The sleeves are independent: either, both or neither can be
  boosted.

  Dip rule (Connors & Alvarez, "Short Term Trading Strategies That Work",
  2008), computed on dividend-adjusted closes:
      enter when  close > SMA(200)  and  RSI(2) < 10
      exit  when  close > SMA(5)

The parameters are the published constants, not fitted to data. Don't tune
them against the backtest. research/qqq_study.py is where the 50/50 version
was chosen from six pre-declared candidates. live.py calls exactly the same
target_weights() that the backtest grades.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Sleeve:
    core: str     # the index ETF the signal is computed on and held by default
    boost: str    # its 2x ETF, held while the dip rule is long
    weight: float


SLEEVES = (
    Sleeve("SPY", "SSO", 0.5),
    Sleeve("QQQ", "QLD", 0.5),
)

RSI_PERIOD = 2
RSI_ENTRY = 10.0
REGIME_SMA_DAYS = 200
EXIT_SMA_DAYS = 5

# Re-size a held position whose share count is this far off target. In the
# backtest and live alike, this is the only rebalance between signal
# changes: it lets the sleeves drift to about 55/45 before pulling them back
# to 50/50. Live, it also tops up a partial fill.
REBALANCE_TOLERANCE = 0.10


def cores() -> list[str]:
    return [s.core for s in SLEEVES]


def universe() -> list[str]:
    return [sym for s in SLEEVES for sym in (s.core, s.boost)]


def rsi(close: pd.Series, period: int = RSI_PERIOD) -> pd.Series:
    """Wilder's RSI."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    out = 100 - 100 / (1 + avg_gain / avg_loss)
    # No losses in the window → inf → 100 already; 0/0 on a flat price → neutral
    return out.fillna(50.0).where(avg_gain.notna())


def dip_position(close: pd.Series) -> pd.Series:
    """True on each date the dip rule is long after that day's close."""
    regime = close.rolling(REGIME_SMA_DAYS).mean()
    exit_sma = close.rolling(EXIT_SMA_DAYS).mean()
    r = rsi(close)

    entry = ((close > regime) & (r < RSI_ENTRY)).to_numpy()
    exit_ = (close > exit_sma).to_numpy()

    held = []
    in_pos = False
    for e, x in zip(entry, exit_):
        if in_pos and x:
            in_pos = False
        elif not in_pos and e:
            in_pos = True
        held.append(in_pos)
    return pd.Series(held, index=close.index)


def target_weights(core_closes: pd.DataFrame) -> pd.DataFrame:
    """
    Per-symbol portfolio weight to hold after each date's close.

    core_closes has one column per sleeve core (SPY, QQQ), on a shared
    date index.
    """
    out = {}
    for s in SLEEVES:
        boost = dip_position(core_closes[s.core]).astype(float)
        out[s.core] = s.weight * (1.0 - boost)
        out[s.boost] = s.weight * boost
    return pd.DataFrame(out, index=core_closes.index)[universe()]
