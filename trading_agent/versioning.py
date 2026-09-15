"""Version stamps recorded on every signal and rejected opportunity.

Every code change that can affect decisions must bump the relevant
version here, so historical records stay interpretable: which strategy,
config, prompts, indicators and risk engine produced this signal?
(spec §42 — versioning)
"""
from __future__ import annotations

# Bump to "INTELLIGENCE_V2" once the upgrade is validated out-of-sample;
# the LEGACY_BASELINE must remain runnable for A/B comparison (§40).
STRATEGY_VERSION = "LEGACY_BASELINE"
CONFIG_VERSION = "1"
PROMPT_VERSION = "TECH_V1/REGIME_V1/SENT_V1"
INDICATOR_VERSION = "1"
RISK_ENGINE_VERSION = "1"


def version_stamp() -> dict:
    """All version identifiers for one decision record."""
    return {
        "strategy_version": STRATEGY_VERSION,
        "config_version": CONFIG_VERSION,
        "prompt_version": PROMPT_VERSION,
        "indicator_version": INDICATOR_VERSION,
        "risk_engine_version": RISK_ENGINE_VERSION,
    }
