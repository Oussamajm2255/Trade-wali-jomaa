"""Feature ablation (V-MONSTER §72) + realistic backtest execution (§69/§70).

The guarantees under test:
- Feature toggles: each `feature_*_enabled` switch removes its group's
  inputs from the canonical snapshot — empty structure map, no DXY
  gauge/context, no VWAP, no liquidity map, no speed state — and the
  defaults keep everything the live pipeline computes.
- Realistic fills: the fill is delayed by the signal's stored Phase G
  reaction window (configured budget as fallback), priced with the
  pace-based drift projection plus slippage, and deferred to the first
  candle that opens after the fill — never ahead of the reaction clock.
- The ablation harness: WITH vs WITHOUT runs over the same window on
  isolated databases, verdicts restricted to the §41 rule set.
"""

from __future__ import annotations

import pandas as pd
import pytest

from analytics_helpers import WARMUP, dxy_frame, gold_frame, settings
from test_backtest import _dxy_frame as backtest_dxy, _gold_frame as backtest_gold
from test_snapshot import FakeMarket, fresh_gauge
from trading_agent.analytics.ablation import FEATURE_GROUPS, run_ablation
from trading_agent.backtest.engine import BacktestEngine, reaction_fill
from trading_agent.config import Settings
from trading_agent.data.snapshot import build_market_snapshot
from trading_agent.schema.types import Side, SignalProposal
from trading_agent.store.db import session_scope
from trading_agent.store.models import Proposal, SignalRecord

TS = pd.Timestamp("2026-01-01 00:00", tz="UTC")


def _proposal(side: Side = Side.LONG, entry: float = 3000.0) -> SignalProposal:
    return SignalProposal(
        id="p1",
        signal_id="s1",
        symbol="XAUUSD",
        timeframe="15m",
        side=side,
        confidence=0.7,
        entry=entry,
        stop=2990.0,
        target=3050.0,
        size=10.0,
        risk_amount=100.0,
        expected_rr=2.0,
        rationale="ablation test",
        evidence={},
        model="test",
    )


def _signal_ts(gold: pd.DataFrame, signal_id: str) -> pd.Timestamp:
    """Replay timestamp of a signal; SQLite returns naive datetimes."""
    with session_scope() as session:
        row = session.get(SignalRecord, signal_id)
    assert row is not None
    return pd.Timestamp(row.ts).tz_localize("UTC")


def _utc(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    return ts if ts.tzinfo is not None else ts.tz_localize("UTC")


# ----------------------------------------------------------- feature toggles


def test_feature_groups_cover_the_planned_axes() -> None:
    assert set(FEATURE_GROUPS) == {"smc", "dxy", "vwap", "liquidity", "speed"}
    for group, profile in FEATURE_GROUPS.items():
        assert list(profile) == [f"feature_{group}_enabled"]
        assert profile[f"feature_{group}_enabled"] is False


def test_snapshot_toggles_remove_group_inputs() -> None:
    snap = build_market_snapshot(
        FakeMarket(gauge=fresh_gauge()),
        "XAUUSD",
        Settings(
            telegram_bot_token="",
            telegram_chat_id="",
            feature_smc_enabled=False,
            feature_dxy_enabled=False,
            feature_vwap_enabled=False,
            feature_liquidity_enabled=False,
            feature_speed_enabled=False,
        ),
        "15m",
    )
    assert snap.structure == {}
    assert snap.dxy is None
    assert snap.dxy_context is None
    assert snap.liquidity == {}
    assert snap.vwap == {}
    assert snap.speed == {}


def test_snapshot_defaults_keep_all_features() -> None:
    gauge = fresh_gauge()
    snap = build_market_snapshot(
        FakeMarket(gauge=gauge),
        "XAUUSD",
        Settings(telegram_bot_token="", telegram_chat_id=""),
        "15m",
    )
    assert snap.structure["timeframe"] == "15m"
    assert snap.dxy["value"] == gauge["value"]
    assert snap.dxy["kind"] == gauge["kind"]
    assert snap.dxy_context is not None
    assert snap.liquidity["levels"]
    assert snap.vwap["available"] is True
    assert snap.speed["state"] is not None


# ------------------------------------------------------------- reaction fill


def test_reaction_fill_uses_stored_timing() -> None:
    fill = reaction_fill(
        _proposal(Side.LONG, 3000.0),
        TS,
        {"reaction_s": 60.0, "pace_per_minute": 0.5},
        settings(),
    )
    assert fill.fill_ts == TS + pd.Timedelta(seconds=60)
    assert fill.drift_price == pytest.approx(3000.0 + 0.5 * 60 / 60)


def test_reaction_fill_short_side_projects_downward() -> None:
    fill = reaction_fill(
        _proposal(Side.SHORT, 3000.0),
        TS,
        {"reaction_s": 30.0, "pace_per_minute": 2.0},
        settings(),
    )
    assert fill.drift_price == pytest.approx(3000.0 - 2.0 * 30 / 60)


def test_reaction_fill_falls_back_to_configured_budget() -> None:
    s = settings(user_reaction_seconds=90.0, telegram_latency_s=10.0)
    fill = reaction_fill(_proposal(), TS, None, s)
    assert fill.fill_ts == TS + pd.Timedelta(seconds=100)
    assert fill.drift_price == 3000.0  # no pace -> reaction delay only


def test_reaction_fill_missing_pace_is_reaction_only() -> None:
    fill = reaction_fill(_proposal(), TS, {"reaction_s": 45.0}, settings())
    assert fill.fill_ts == TS + pd.Timedelta(seconds=45)
    assert fill.drift_price == 3000.0


# -------------------------------------------------- realistic engine replay


def _realistic_report(gold: pd.DataFrame, s: Settings, tmp_path, db_name: str):
    db = f"sqlite:///{tmp_path.as_posix()}/{db_name}"
    return BacktestEngine(
        s, {"15m": gold}, symbol="XAUUSD", timeframe="15m",
        dxy_frames=backtest_dxy(), db_url=db, warmup=WARMUP,
    ).run()


def test_realistic_fill_delays_by_reaction_window(tmp_path) -> None:
    gold = backtest_gold(n=450)
    s = settings(backtest_realistic_execution=True)
    report = _realistic_report(gold, s, tmp_path, "real.db")
    assert report.stats["trades"] >= 1
    trade = report.trades[0]
    sig_ts = _signal_ts(gold, trade.signal_id)
    expected = sig_ts + pd.Timedelta(
        seconds=s.user_reaction_seconds + s.telegram_latency_s
    )
    # The reaction window elapsed at exactly the scheduled instant — the
    # EMA fallback in the isolated DB equals the configured budget.
    assert _utc(trade.opened_at) == expected


def test_realistic_fill_prices_drift_plus_slippage(tmp_path) -> None:
    gold = backtest_gold(n=450)
    s = settings(backtest_realistic_execution=True)
    report = _realistic_report(gold, s, tmp_path, "drift.db")
    trade = report.trades[0]
    with session_scope() as session:
        record = session.get(SignalRecord, trade.signal_id)
        timing = (record.fusion or {}).get("timing") or {}
        proposal = session.get(Proposal, record.proposal_id)
    pace = timing.get("pace_per_minute")
    if isinstance(timing.get("reaction_s"), (int, float)):
        reaction_s = float(timing["reaction_s"])
    else:
        reaction_s = s.user_reaction_seconds + s.telegram_latency_s
    drift = float(pace) * reaction_s / 60 if isinstance(pace, (int, float)) else 0.0
    sign = 1.0 if trade.side == "long" else -1.0
    base = float(proposal.entry) + sign * drift
    expected = base * (1 + sign * s.slippage)  # spread_pct = 0
    assert trade.entry == pytest.approx(expected, abs=1e-6)


def test_realistic_fill_defers_when_reaction_crosses_candle(tmp_path) -> None:
    gold = backtest_gold(n=450)
    s = settings(
        backtest_realistic_execution=True,
        user_reaction_seconds=1200.0,  # 20 min > 15m candle
        telegram_latency_s=0.0,
        timing_gate_enabled=False,  # TOO_LATE must not cull the sample
    )
    report = _realistic_report(gold, s, tmp_path, "defer.db")
    assert report.stats["trades"] >= 1
    trade = report.trades[0]
    sig_ts = _signal_ts(gold, trade.signal_id)
    with session_scope() as session:
        record = session.get(SignalRecord, trade.signal_id)
        timing = (record.fusion or {}).get("timing") or {}
    if isinstance(timing.get("reaction_s"), (int, float)):
        reaction_s = float(timing["reaction_s"])
    else:
        # telegram_latency_s=0.0 is falsy, so the pipeline's `or 3.0`
        # fallback applies (same in compute_timing and reaction_fill).
        reaction_s = 1200.0 + 3.0
    opened = _utc(trade.opened_at)
    # Filled after the reaction window, i.e. beyond the next candle's open.
    assert opened == sig_ts + pd.Timedelta(seconds=reaction_s)
    assert opened > sig_ts + pd.Timedelta(minutes=15)


def test_realistic_replay_is_deterministic() -> None:
    gold = backtest_gold(n=450)
    s = settings(backtest_realistic_execution=True)
    # In-memory: every init_engine creates a fresh isolated database, so
    # both runs share the same URL and must be bit-identical.
    def _run():
        return BacktestEngine(
            s, {"15m": gold}, symbol="XAUUSD", timeframe="15m",
            dxy_frames=backtest_dxy(), db_url="sqlite:///:memory:",
            warmup=WARMUP,
        ).run().to_dict()

    assert _run() == _run()


# ---------------------------------------------------------------- ablation


def test_run_ablation_end_to_end(tmp_path) -> None:
    gold = backtest_gold(n=300)
    report = run_ablation(
        settings(),
        {"15m": gold},
        timeframe="15m",
        dxy_frames=dxy_frame(),
        db_url=f"sqlite:///{tmp_path.as_posix()}/abl.db",
        warmup=WARMUP,
        min_trades=0,
    )
    assert set(report.results) == set(FEATURE_GROUPS)
    base = report.baseline
    assert base.group == "with_all_features"
    assert base.verdict == "BASELINE"
    for name, r in report.results.items():
        assert r.verdict in {"IMPROVED", "MIXED", "WORSE", "INSUFFICIENT_DATA"}
        # Same window, isolated databases.
        assert r.report.start_ts == base.report.start_ts
        assert r.report.end_ts == base.report.end_ts
        assert r.report.db_url != base.report.db_url
        assert r.report.db_url.endswith(f"_abl_{name}.db")
    d = report.to_dict()
    assert d["results"]["smc"]["verdict"] == report.results["smc"].verdict
    assert d["baseline"]["stats"]["trades"] == base.report.stats["trades"]


def test_run_ablation_verdicts_on_tradeable_window(tmp_path) -> None:
    gold = backtest_gold(n=400)
    report = run_ablation(
        settings(),
        {"15m": gold},
        groups=["smc"],
        timeframe="15m",
        dxy_frames=dxy_frame(),
        db_url=f"sqlite:///{tmp_path.as_posix()}/trade.db",
        warmup=WARMUP,
        min_trades=0,
    )
    assert report.baseline.report.stats["trades"] >= 1
    result = report.results["smc"]
    assert result.verdict in {"IMPROVED", "MIXED", "WORSE", "INSUFFICIENT_DATA"}


def test_run_ablation_subset_is_deterministic(tmp_path) -> None:
    gold = backtest_gold(n=350)
    base = settings()
    url = f"sqlite:///{tmp_path.as_posix()}/sub.db"
    first = run_ablation(
        base, {"15m": gold}, groups=["smc", "dxy"], timeframe="15m",
        db_url=url, warmup=WARMUP, min_trades=0,
    ).to_dict()
    second = run_ablation(
        base, {"15m": gold}, groups=["smc", "dxy"], timeframe="15m",
        db_url=url, warmup=WARMUP, min_trades=0,
    ).to_dict()
    assert first == second


def test_run_ablation_rejects_unknown_group(tmp_path) -> None:
    with pytest.raises(ValueError, match="unknown feature groups"):
        run_ablation(
            settings(),
            {"15m": gold_frame(n=300)},
            groups=["smc", "nonsense"],
            warmup=WARMUP,
            db_url=f"sqlite:///{tmp_path.as_posix()}/bad.db",
        )


# ----------------------------------------------------------- sample size (§36)


def test_required_trades_textbook_value() -> None:
    from trading_agent.analytics.sample_size import required_trades

    # Two-sided, alpha .05, power .8, effect = 1 stdev: the textbook
    # two-sample answer is 16 per side.
    assert required_trades(1.0) == 16


def test_required_trades_scales_with_effect_and_stdev() -> None:
    from trading_agent.analytics.sample_size import required_trades

    # Half the effect in the same units -> ~4x the sample.
    small = required_trades(0.5)
    assert small == pytest.approx(4 * required_trades(1.0), abs=1)
    # Effect = 2 stdevs needs a quarter of the sample.
    assert required_trades(2.0) == 4


def test_required_trades_rejects_invalid_inputs() -> None:
    from trading_agent.analytics.sample_size import required_trades

    with pytest.raises(ValueError):
        required_trades(0.0)
    with pytest.raises(ValueError):
        required_trades(-0.5)
    with pytest.raises(ValueError):
        required_trades(1.0, stdev_r=0.0)
    with pytest.raises(ValueError):
        required_trades(1.0, power=1.0)
    with pytest.raises(ValueError):
        required_trades(1.0, alpha=0.0)


def test_detectable_effect_inverts_required_trades() -> None:
    from trading_agent.analytics.sample_size import detectable_effect, required_trades

    n = required_trades(0.4)
    # At the required N (rounded up) the detectable effect is at most
    # the asked one — the ceiling only makes the sample more sensitive.
    assert detectable_effect(n) == pytest.approx(0.4, rel=0.01)
    assert detectable_effect(n) <= 0.4
    # Smaller sample -> only larger effects are detectable.
    assert detectable_effect(10) > detectable_effect(100)


def test_detectable_effect_rejects_invalid_inputs() -> None:
    from trading_agent.analytics.sample_size import detectable_effect

    with pytest.raises(ValueError):
        detectable_effect(0)
    with pytest.raises(ValueError):
        detectable_effect(10, stdev_r=-1.0)
    with pytest.raises(ValueError):
        detectable_effect(10, power=0.0)
