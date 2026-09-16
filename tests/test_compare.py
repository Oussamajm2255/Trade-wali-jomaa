"""§40/§41 A/B comparison: LEGACY_BASELINE vs INTELLIGENCE_V2 on
identical data; generating fewer trades alone is never an improvement."""

from analytics_helpers import WARMUP, gold_frame, settings
from trading_agent.analytics.compare import (
    STRATEGY_NAMES,
    STRATEGY_PROFILES,
    _sibling_db,
    _verdict,
    run_comparison,
)


def _stats(trades=20, expectancy=0.5, dd=10.0, pf=1.5) -> dict:
    return {
        "trades": trades,
        "win_rate": None,
        "expectancy_r": expectancy,
        "max_drawdown_pct": dd,
        "profit_factor": pf,
    }


def test_profiles_disable_only_post_upgrade_gates():
    legacy = STRATEGY_PROFILES["legacy_baseline"]
    assert legacy["setup_quality_min"] == 0.0
    assert legacy["conflict_block_conflicted"] is False
    assert legacy["statistical_quality_enabled"] is False
    assert legacy["require_statistical_edge"] is False
    # The v1 hard gates (DXY/HTF/min_confidence) are NOT in the profile:
    # they stay identical in both runs.
    assert "min_confidence" not in legacy
    assert "dxy_filter_enabled" not in legacy
    assert STRATEGY_PROFILES["intelligence_v2"] == {}
    assert STRATEGY_NAMES == ("legacy_baseline", "intelligence_v2")


def test_sibling_db_derives_isolated_files():
    a = _sibling_db("sqlite:///backtest.db", "legacy_baseline")
    b = _sibling_db("sqlite:///backtest.db", "intelligence_v2")
    assert a.endswith("backtest_legacy_baseline.db")
    assert b.endswith("backtest_intelligence_v2.db")
    assert a != b


def test_verdict_improved_needs_quality():
    verdict, _ = _verdict(
        _stats(trades=20, expectancy=0.3, dd=12.0, pf=1.2),
        _stats(trades=12, expectancy=0.6, dd=8.0, pf=2.0),
        10,
    )
    assert verdict == "IMPROVED"


def test_verdict_fewer_trades_alone_is_not_improved():
    # Same expectancy, drawdown and PF — fewer trades changes nothing.
    verdict, note = _verdict(
        _stats(trades=50, expectancy=0.5, dd=10.0, pf=1.5),
        _stats(trades=12, expectancy=0.5, dd=10.0, pf=1.5),
        10,
    )
    assert verdict == "MIXED"
    assert "fewer trades" in note


def test_verdict_worse_when_expectancy_and_drawdown_worse():
    verdict, _ = _verdict(
        _stats(expectancy=0.6, dd=8.0, pf=2.0),
        _stats(expectancy=0.4, dd=10.0, pf=1.5),
        10,
    )
    assert verdict == "WORSE"


def test_verdict_improved_but_drawdown_much_worse_is_mixed():
    verdict, _ = _verdict(
        _stats(expectancy=0.3, dd=5.0, pf=1.5),
        _stats(expectancy=0.6, dd=20.0, pf=2.0),
        10,
    )
    assert verdict == "MIXED"


def test_verdict_insufficient_data_under_min_trades():
    verdict, note = _verdict(_stats(trades=5), _stats(trades=20), 10)
    assert verdict == "INSUFFICIENT_DATA"
    assert "10" in note


def test_verdict_insufficient_data_without_resolved_trades():
    verdict, _ = _verdict(_stats(trades=0, expectancy=None), _stats(trades=0, expectancy=None), 0)
    assert verdict == "INSUFFICIENT_DATA"


def test_run_comparison_end_to_end_identical_data(tmp_path):
    gold = gold_frame(n=600)
    report = run_comparison(
        settings(), {"15m": gold}, timeframe="15m",
        db_url=f"sqlite:///{tmp_path.as_posix()}/ab.db",
        warmup=WARMUP, min_trades=0,
    )
    assert set(report.results) == {"legacy_baseline", "intelligence_v2"}
    a = report.results["legacy_baseline"]
    b = report.results["intelligence_v2"]
    assert a.verdict == "BASELINE"
    assert b.verdict in {"IMPROVED", "MIXED", "WORSE", "INSUFFICIENT_DATA"}
    # Same data window for both sides.
    assert a.report.start_ts == b.report.start_ts
    assert a.report.end_ts == b.report.end_ts
    assert a.report.candles == b.report.candles
    # Isolated databases: no shared state.
    assert a.report.db_url != b.report.db_url
    d = report.to_dict()
    assert d["results"]["legacy_baseline"]["stats"]["trades"] == a.report.stats["trades"]
    assert d["results"]["intelligence_v2"]["verdict"] == b.verdict


def test_run_comparison_is_deterministic(tmp_path):
    gold = gold_frame(n=500)
    base = settings()
    url = f"sqlite:///{tmp_path.as_posix()}/ab.db"
    first = run_comparison(base, {"15m": gold}, timeframe="15m", db_url=url,
                           warmup=WARMUP, min_trades=0).to_dict()
    second = run_comparison(base, {"15m": gold}, timeframe="15m", db_url=url,
                            warmup=WARMUP, min_trades=0).to_dict()
    assert first == second
