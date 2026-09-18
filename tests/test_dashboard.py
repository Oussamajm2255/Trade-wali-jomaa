"""§45 intelligence dashboard + §46 confidence buckets."""

from datetime import datetime, timedelta, timezone

from trading_agent.analytics.stats import confidence_buckets, resolved_signals
from trading_agent.dashboard.report import build_dashboard_html, collect, render_html
from trading_agent.store import actions
from trading_agent.store.db import session_scope


def _signal(signal_id, ts, outcome, r, conf, decision="proposal", **overrides) -> dict:
    return {
        "signal_id": signal_id,
        "ts": ts,
        "symbol": "XAUUSD",
        "timeframe": "15m",
        "strategy_version": "LEGACY_BASELINE",
        "config_version": "v1",
        "prompt_version": "p1",
        "market_snapshot": {
            "symbol": "XAUUSD",
            "timeframe": "15m",
            "data_quality": "good",
            "price": 4350.0,
            "regime": {"regime": "trend_up"},
            "session_context": {"session": "LONDON"},
            "alignment": {"label": "aligned"},
            "dxy_gauge": {"value": 60, "classification": "Bullish (USD weak)"},
            "mtf_biases": {"4h": {"bias": "long"}},
        },
        "ai_outputs": {},
        "fusion": {"raw_confidence": conf, "direction_score": conf},
        "setup_quality": {"score": 0.7},
        "conflicts": {"state": "ALIGNED", "conflicts": []},
        "gates": [{"gate": "risk", "status": "pass", "detail": ""}],
        "final_decision": decision,
        "outcome": outcome,
        "r_multiple": r,
        **overrides,
    }


def _rows():
    with session_scope() as session:
        return resolved_signals(session, limit=5000)


# ------------------------------------------------------------- §46 buckets


def test_bucket_boundaries_are_exact():
    base = datetime(2026, 9, 16, 10, 0)
    specs = [
        ("a", 0.50, "WIN", 1.0),   # lower bound of first bucket
        ("b", 0.549, "WIN", 1.0),  # just inside first bucket
        ("c", 0.55, "WIN", 1.0),   # second bucket
        ("d", 0.799, "WIN", 1.0),
        ("e", 0.80, "LOSS", -1.0),  # 0.80+ bucket
        ("f", 0.99, "LOSS", -1.0),
    ]
    for i, (sid, conf, outcome, r) in enumerate(specs):
        actions.record_signal(_signal(sid, base + timedelta(minutes=i), outcome, r, conf))
    buckets = {b["bucket"]: b for b in confidence_buckets(_rows(), min_sample=1)}
    assert buckets["0.50–0.55"]["trades"] == 2
    assert buckets["0.55–0.60"]["trades"] == 1
    assert buckets["0.60–0.65"]["trades"] == 0
    assert buckets["0.65–0.70"]["trades"] == 0
    assert buckets["0.70–0.80"]["trades"] == 1
    assert buckets["0.80+"]["trades"] == 2


def test_bucket_stats_win_rate_expectancy_avg_r():
    base = datetime(2026, 9, 16, 10, 0)
    actions.record_signal(_signal("w1", base, "WIN", 2.0, 0.56))
    actions.record_signal(_signal("l1", base + timedelta(minutes=1), "LOSS", -1.0, 0.59))
    actions.record_signal(_signal("w2", base + timedelta(minutes=2), "WIN", 0.5, 0.56))
    buckets = {b["bucket"]: b for b in confidence_buckets(_rows(), min_sample=5)}
    bucket = buckets["0.55–0.60"]
    assert bucket["trades"] == 3
    assert bucket["wins"] == 2
    assert bucket["win_rate"] == round(2 / 3, 4)
    assert bucket["expectancy_r"] == round((2.0 - 1.0 + 0.5) / 3, 4)
    assert bucket["avg_r"] == bucket["expectancy_r"]
    assert bucket["sufficient"] is False  # 3 < min_sample 5


def test_bucket_sufficiency_flag():
    base = datetime(2026, 9, 16, 10, 0)
    for i in range(3):
        actions.record_signal(
            _signal(f"w{i}", base + timedelta(minutes=i), "WIN", 1.0, 0.61)
        )
    buckets = {b["bucket"]: b for b in confidence_buckets(_rows(), min_sample=3)}
    assert buckets["0.60–0.65"]["sufficient"] is True
    assert buckets["0.70–0.80"]["trades"] == 0
    assert buckets["0.70–0.80"]["sufficient"] is False


def test_buckets_ignore_missing_confidence():
    base = datetime(2026, 9, 16, 10, 0)
    actions.record_signal(_signal("n1", base, "WIN", 1.0, conf=None))
    assert all(b["trades"] == 0 for b in confidence_buckets(_rows(), min_sample=1))


def test_buckets_empty_population():
    assert all(b["trades"] == 0 for b in confidence_buckets(_rows(), min_sample=1))


# ---------------------------------------------------------------- dashboard


def test_collect_counts_activity_and_overall():
    base = datetime(2026, 9, 16, 10, 0)
    today = datetime.now(timezone.utc).replace(microsecond=0)
    actions.record_signal(
        _signal("win", today - timedelta(minutes=5), "WIN", 2.0, 0.7,
                decision="proposal")
    )
    actions.record_signal(
        _signal("loss", today - timedelta(minutes=4), "LOSS", -1.0, 0.55,
                decision="proposal")
    )
    actions.record_signal(
        _signal("rej", today - timedelta(minutes=3), None, None, None,
                decision="rejected", no_trade_reason="DXY_FILTER",
                decision_reason="dollar trop fort", sl=None)
    )
    with session_scope() as session:
        data = collect(session)
    assert data["overall"]["trades"] == 2
    assert data["overall"]["wins"] == 1
    assert data["overall"]["win_rate"] == 0.5
    assert data["today"]["proposals"] == 2
    assert data["today"]["rejected"] == 1
    assert data["rejection_reasons"] == [{"reason": "DXY_FILTER", "count": 1}]
    assert data["current"]["symbol"] == "XAUUSD"
    assert data["current"]["regime"] == "Tendance haussière"
    assert data["current"]["session"] == "Londres"
    assert len(data["recent_signals"]) == 3
    assert data["confidence_buckets"][0]["bucket"] == "0.50–0.55"


def test_collect_empty_database():
    with session_scope() as session:
        data = collect(session)
    assert data["overall"]["trades"] == 0
    assert data["current"]["price"] is None
    assert data["recent_signals"] == []
    assert data["rejection_reasons"] == []


def test_render_html_is_self_contained():
    base = datetime.now(timezone.utc).replace(microsecond=0)
    actions.record_signal(
        _signal("win", base, "WIN", 1.0, 0.62, decision="proposal")
    )
    with session_scope() as session:
        html_text = build_dashboard_html(session)
    assert html_text.startswith("<!DOCTYPE html>")
    assert "<style>" in html_text
    assert "Tableau de bord" in html_text
    assert "Performance globale" in html_text
    assert "Confiance par tranche" in html_text
    assert "Distribution de la confiance" in html_text
    assert "Raisons de rejet" in html_text
    assert "Performance par :" in html_text
    assert "LEGACY_BASELINE" in html_text
    # No external network references.
    assert "http://" not in html_text
    assert "https://" not in html_text
    assert "<script" not in html_text


def test_render_html_empty_database_still_renders():
    with session_scope() as session:
        html_text = build_dashboard_html(session)
    assert "<!DOCTYPE html>" in html_text
    assert "Aucun signal." in html_text
    assert "Aucun rejet enregistré." in html_text


def test_render_html_escapes_markup():
    data = {
        "generated_at": "2026-09-16",
        "versions": {"strategy_version": "LEGACY_BASELINE", "config_version": "v1",
                     "prompt_version": "p1"},
        "current": {"ts": None, "symbol": "<b>X</b>", "timeframe": None, "decision": None,
                    "price": None, "regime": "—", "session": "—", "alignment": "—",
                    "dxy_gauge": {}, "dxy_context": {}, "confidence": None,
                    "calibrated": None, "setup_quality": None, "mtf_biases": {}},
        "today": {"proposals": 0, "rejected": 0, "trades": 0, "actionable_rate": None},
        "overall": {"trades": 0, "wins": 0, "losses": 0, "win_rate": None,
                    "expectancy_r": None, "avg_r": None, "profit_factor": None,
                    "total_r": 0.0, "max_drawdown_r": 0.0},
        "resolved_count": 0,
        "records_analysed": 0,
        "confidence_buckets": [],
        "confidence_histogram": {},
        "setup_quality_histogram": {},
        "setup_quality_performance": [],
        "rejection_reasons": [],
        "breakdowns": {},
        "recent_signals": [],
        "rejection_quality": {"correct": 0, "wrong": 0, "inconclusive": 0,
                               "resolved": 0, "correct_rate": None},
    }
    html_text = render_html(data)
    assert "<b>X</b>" not in html_text
    assert "&lt;b&gt;X&lt;/b&gt;" in html_text
