"""Market shock detection (spec §44): NORMAL / VOLATILITY_EXPANSION /
SHOCK classification, the SHOCK risk gate and the persisted cooldown."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from trading_agent.config import Settings
from trading_agent.data.shock import ShockState, detect_shock
from trading_agent.data.snapshot import build_market_snapshot
from trading_agent.fusion.types import NoTradeReason
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, Side
from trading_agent.store.db import session_scope
from trading_agent.store.models import RiskState


def _frame(n: int = 100, range_: float = 1.0, volume: float = 100.0) -> pd.DataFrame:
    """n flat candles; the caller mutates the LAST candle to inject a spike."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="15min", tz="UTC")
    close = 4350.0 + pd.Series(range(n), dtype=float).to_numpy() * 0.01
    df = pd.DataFrame(
        {
            "open": close - range_ / 2,
            "high": close + range_ / 2,
            "low": close - range_ / 2,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )
    return df


def _spike(df: pd.DataFrame, range_: float, body: float = 0.0, gap: float = 0.0) -> pd.DataFrame:
    idx = df.index[-1]
    close = df.loc[idx, "close"]
    if gap:
        df.loc[idx, "open"] = close * (1 + gap)  # gap up
    df.loc[idx, "high"] = close + range_ / 2
    df.loc[idx, "low"] = close - range_ / 2
    if body:
        df.loc[idx, "close"] = df.loc[idx, "open"] * (1 + body)
    return df


# ------------------------------------------------------------ detection


def test_normal_flat_market():
    out = detect_shock(_frame(100))
    assert out["state"] == ShockState.NORMAL
    assert out["ratios"]["range"] == pytest.approx(1.0, abs=0.2)


def test_volatility_expansion_when_range_doubles():
    out = detect_shock(_spike(_frame(100), range_=2.0))
    assert out["state"] == ShockState.VOLATILITY_EXPANSION
    assert "range" in out["detail"]


def test_shock_when_range_quadruples():
    out = detect_shock(_spike(_frame(100), range_=4.0))
    assert out["state"] == ShockState.SHOCK
    assert "shock:" in out["detail"]


def test_shock_on_extreme_price_movement():
    out = detect_shock(_spike(_frame(100), range_=1.0, body=0.012))
    assert out["state"] == ShockState.SHOCK
    assert out["movement_pct"] >= 1.0


def test_shock_on_spread_threshold():
    out = detect_shock(
        _frame(100), spread_pct_threshold=0.05, spread=4350.0 * 0.001
    )  # 0.1% > 0.05%
    assert out["state"] == ShockState.SHOCK
    assert "spread" in out["detail"]


def test_volume_axis_contributes():
    df = _frame(100)
    df.iloc[-1, df.columns.get_loc("volume")] = 100.0 * 4
    out = detect_shock(df)
    assert out["state"] == ShockState.SHOCK
    assert out["ratios"]["volume"] >= 3.0


def test_volume_axis_skipped_when_untrusted():
    """Proxy data (PAXG token): a volume spike alone must not shock."""
    df = _frame(100)
    df.iloc[-1, df.columns.get_loc("volume")] = 100.0 * 4
    out = detect_shock(df, trust_volume=False)
    assert out["state"] == ShockState.NORMAL
    assert "volume" not in out["ratios"]
    assert "volume skipped: proxy data" in out["detail"]


def test_untrusted_volume_but_range_spike_still_shocks():
    """Price-based axes stay active even when volume is untrusted."""
    df = _frame(100)
    df.iloc[-1, df.columns.get_loc("volume")] = 100.0 * 4
    out = detect_shock(_spike(df, range_=4.0), trust_volume=False)
    assert out["state"] == ShockState.SHOCK
    assert out["ratios"]["range"] >= 3.0
    assert "volume skipped: proxy data" in out["detail"]


def test_insufficient_history_never_shocks():
    out = detect_shock(_spike(_frame(20), range_=50.0), lookback=60)
    assert out["state"] == ShockState.NORMAL
    assert "insufficient" in out["detail"]


def test_zero_baseline_range_is_no_signal():
    # A truly flat baseline (TR = 0 everywhere): the spike's ratios have
    # no baseline to compare against and must NOT shock (fail open).
    df = _frame(100, range_=0.0)
    df["open"] = 4350.0
    df["close"] = 4350.0
    out = detect_shock(_spike(df, range_=0.5))
    assert out["state"] == ShockState.NORMAL


def test_deterministic_repeat():
    df = _spike(_frame(100), range_=4.0)
    assert detect_shock(df) == detect_shock(df)


# ------------------------------------------------------------ risk gate


def _settings(**overrides) -> Settings:
    return Settings(
        min_confidence=0.55,
        setup_quality_min=0.0,
        telegram_bot_token="",
        telegram_chat_id="",
        **overrides,
    )


def _eval(settings: Settings, shock_context: dict, now: datetime | None = None):
    engine = RiskEngine(settings)
    gates: list[dict] = []
    result = engine.evaluate(
        symbol="XAUUSD",
        timeframe="15m",
        side=Side.LONG,
        confidence=0.7,
        price=4350.0,
        atr=12.5,
        verdicts={},
        gauge=None,
        context={"shock_context": shock_context},
        trail=gates,
        now=now,
    )
    return result, gates


def _set_last_shock_ts(ts: datetime | None) -> None:
    with session_scope() as session:
        state = session.get(RiskState, 1)
        state.last_shock_ts = ts
        session.commit()


def test_shock_gate_blocks_new_entries():
    result, gates = _eval(_settings(), {"state": ShockState.SHOCK, "detail": "shock: range 4.0x"})
    assert isinstance(result, Rejection)
    assert result.no_trade_reason == NoTradeReason.SHOCK.value
    shock_gate = next(g for g in gates if g["gate"] == "shock")
    assert shock_gate["status"] == "reject"


def test_shock_persists_cooldown_anchor():
    now = datetime.now(timezone.utc)
    _eval(_settings(), {"state": ShockState.SHOCK, "detail": "shock: range 4.0x"}, now=now)
    with session_scope() as session:
        assert session.get(RiskState, 1).last_shock_ts is not None


def test_shock_cooldown_blocks_while_active():
    now = datetime.now(timezone.utc)
    _set_last_shock_ts(now - timedelta(minutes=10))
    result, gates = _eval(
        _settings(shock_cooldown_minutes=30),
        {"state": ShockState.NORMAL, "detail": "normal volatility"},
        now=now,
    )
    assert isinstance(result, Rejection)
    assert "cooldown" in result.reason
    assert next(g for g in gates if g["gate"] == "shock")["status"] == "reject"


def test_shock_cooldown_expires():
    now = datetime.now(timezone.utc)
    _set_last_shock_ts(now - timedelta(minutes=45))
    result, _ = _eval(
        _settings(shock_cooldown_minutes=30),
        {"state": ShockState.NORMAL, "detail": "normal volatility"},
        now=now,
    )
    assert not isinstance(result, Rejection)


def test_shock_gate_disabled_never_blocks():
    result, gates = _eval(
        _settings(shock_enabled=False),
        {"state": ShockState.SHOCK, "detail": "shock: range 4.0x"},
    )
    assert not isinstance(result, Rejection)
    assert not any(g["gate"] == "shock" for g in gates)


def test_shock_block_can_be_softened_to_cooldown_only():
    # shock_block_new_entries=False: the SHOCK candle itself is a warning,
    # the anchor is still stored and blocks the next cycle via cooldown.
    now = datetime.now(timezone.utc)
    result, gates = _eval(
        _settings(shock_block_new_entries=False, shock_cooldown_minutes=30),
        {"state": ShockState.SHOCK, "detail": "shock: range 4.0x"},
        now=now,
    )
    assert not isinstance(result, Rejection)
    assert next(g for g in gates if g["gate"] == "shock")["status"] == "warning"
    with session_scope() as session:
        assert session.get(RiskState, 1).last_shock_ts is not None


def test_volatility_expansion_is_warning_only():
    result, gates = _eval(
        _settings(),
        {"state": ShockState.VOLATILITY_EXPANSION, "detail": "volatility expansion: range 2.0x"},
    )
    assert not isinstance(result, Rejection)
    assert next(g for g in gates if g["gate"] == "shock")["status"] == "warning"


# ----------------------------------------------------------- snapshot wiring


class _ProxyMarket:
    """Market whose candles come from a proxy feed with a volume spike."""

    last_source = "PAXG/USDT proxy (yfinance unavailable)"

    def __init__(self) -> None:
        self.calendar_provider = None

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        n = 300
        idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=n, freq="15min", tz="UTC")
        close = 4350.0 + pd.Series(range(n), dtype=float).to_numpy() * 0.01
        df = pd.DataFrame(
            {"open": close - 0.5, "high": close + 0.5, "low": close - 0.5,
             "close": close, "volume": 100.0},
            index=idx,
        )
        df.iloc[-1, df.columns.get_loc("volume")] = 100.0 * 4  # would SHOCK if trusted
        return df


def test_snapshot_disables_volume_axis_on_proxy_data():
    """The snapshot flags proxy feeds so a token volume spike never
    triggers a false SHOCK (production incident: PAXG volume 9.5x)."""
    settings = Settings(
        snapshot_timeframes=["1h"], htf_timeframe="4h", htf_bias_filter_enabled=False,
        telegram_bot_token="", telegram_chat_id="",
    )
    snap = build_market_snapshot(_ProxyMarket(), "XAUUSD", settings, "15m")
    assert "proxy" in snap.data_source.lower()
    assert snap.shock_context["state"] == ShockState.NORMAL
    assert "volume skipped: proxy data" in snap.shock_context["detail"]
    assert "volume" not in snap.shock_context["ratios"]
