"""§47 feature contribution: WHAT SUPPORTS / CONTRADICTS / INVALIDATES
a signal — derived from the stored record only, never from the LLM."""

from trading_agent.analytics.contribution import FeatureContribution, feature_contribution


def _record(**overrides) -> dict:
    record = {
        "market_snapshot": {
            "alignment": {"label": "aligned"},
            "structure": {"bos": True, "choch": True, "fvgs": [1], "sweeps": True},
            "dxy_context": {"classification": "Bullish (USD weak)"},
        },
        "setup_quality": {
            "score": 0.7,
            "components": {
                "mtf": 0.8,
                "structure": 0.7,
                "regime": 0.6,
                "dxy": 0.9,
                "volatility": 0.3,
                "session": 0.4,
                "risk_reward": 0.8,
            },
        },
        "conflicts": {"state": "ALIGNED", "conflicts": [], "conflict_score": 0.0},
        "sl": 4300.0,
    }
    record.update(overrides)
    return record


def test_supporting_strong_components():
    c = feature_contribution(_record())
    assert "alignement MTF fort (0.80)" in c.supporting
    assert "contexte DXY fort (0.90)" in c.supporting
    assert "ratio risque/rendement fort (0.80)" in c.supporting
    # threshold boundary: exactly 0.6 supports
    assert any("régime fort (0.60)" == s for s in c.supporting)


def test_contradicting_weak_components():
    c = feature_contribution(_record())
    assert "volatilité faible (0.30)" in c.contradicting
    assert "session faible (0.40)" in c.contradicting


def test_middle_scores_are_neither():
    c = feature_contribution(
        _record(
            market_snapshot={},
            setup_quality={"score": 0.5, "components": {"mtf": 0.5, "dxy": 0.55}},
        )
    )
    assert c.supporting == []
    assert c.contradicting == []


def test_alignment_and_structure_and_dxy_listed():
    c = feature_contribution(_record())
    assert "alignement aligned" in c.supporting
    assert "structure: BOS, CHoCH, FVG, balayage de liquidité" in c.supporting
    assert "DXY: Bullish (USD weak)" in c.supporting


def test_gauge_fallback_for_dxy_classification():
    record = _record()
    record["market_snapshot"].pop("dxy_context")
    record["market_snapshot"]["dxy_gauge"] = {"classification": "Neutral"}
    c = feature_contribution(record)
    assert "DXY: Neutral" in c.supporting


def test_conflicts_append_to_contradicting():
    c = feature_contribution(
        _record(
            conflicts={
                "state": "CONFLICTED",
                "conflicts": [
                    {"axis": "mtf", "no_trade_reason": "MTF_CONFLICT", "detail": "4h baissière contre 15m haussière"},
                    {"axis": "dxy", "no_trade_reason": "DXY_CONFLICT", "detail": "dollar fort contre un LONG"},
                ],
            },
        )
    )
    assert "4h baissière contre 15m haussière" in c.contradicting
    assert "dollar fort contre un LONG" in c.contradicting


def test_invalidation_is_the_stop():
    c = feature_contribution(_record(sl=4300.0))
    assert "setup invalidé si le prix clôture au-delà du SL (4,300)" in c.invalidation


def test_invalidation_rejected_before_sizing():
    record = _record(sl=None)
    c = feature_contribution(record)
    assert c.invalidation == "pas de niveau calculé — rejeté avant sizing"


def test_no_invalidation_when_clean_and_no_stop():
    record = _record(sl=None, setup_quality={"score": 0.7, "components": {"mtf": 0.8}})
    assert feature_contribution(record).invalidation == ""


def test_empty_record_yields_empty_contribution():
    c = feature_contribution({})
    assert c.supporting == []
    assert c.contradicting == []
    assert c.invalidation == ""


def test_to_dict_shape():
    c = feature_contribution(_record())
    d = c.to_dict()
    assert set(d) == {"supporting", "contradicting", "invalidation"}
    assert isinstance(d["supporting"], list)
    assert isinstance(d["invalidation"], str)


def test_returns_feature_contribution_type():
    assert isinstance(feature_contribution(_record()), FeatureContribution)


def test_non_numeric_components_ignored():
    c = feature_contribution(
        _record(
            market_snapshot={},
            setup_quality={"score": 0.7, "components": {"mtf": "high", "dxy": None}},
        )
    )
    assert c.supporting == []
    assert c.contradicting == []
