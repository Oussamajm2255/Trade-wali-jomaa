"""Intelligence dashboard (spec §45): one self-contained HTML report —
current market read, today's activity, performance, distributions,
rejection reasons and per-dimension breakdowns.

Everything comes from the local database (spec §22); no external
assets, no network, no JavaScript. The report is static HTML readable
on a normal desktop screen. Confidence analytics follow §46: only
buckets with a sufficient sample may be interpreted.
"""

from __future__ import annotations

import html
from collections import Counter
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from trading_agent.analytics.stats import (
    _CONFIDENCE_BUCKETS,
    breakdown,
    compute_trade_stats,
    confidence_buckets,
    resolved_signals,
)
from trading_agent.store.models import Position, SignalRecord
from trading_agent.versioning import version_stamp

# Setup-quality display tiers (spec §45: "performance by setup quality").
_SQ_TIERS = [
    ("0.00–0.40", 0.0, 0.40),
    ("0.40–0.60", 0.40, 0.60),
    ("0.60–0.80", 0.60, 0.80),
    ("0.80–1.00", 0.80, 1.01),
]

_SQ_HIST_BINS = [
    (f"{i / 5:.1f}–{(i + 1) / 5:.1f}", i / 5, (i + 1) / 5 if i < 4 else 1.01)
    for i in range(5)
]

_DIM_LABELS = {
    "side": "Direction",
    "regime": "Régime",
    "session": "Session",
    "alignment": "Alignement MTF",
    "structure": "Structure",
    "dxy": "Contexte DXY",
}

_REGIME_LABELS = {
    "trend_up": "Tendance haussière",
    "trend_down": "Tendance baissière",
    "range": "Range",
    "transition": "Transition",
    "high_volatility": "Volatilité élevée",
    "low_volatility": "Volatilité basse",
}
_SESSION_LABELS = {
    "ASIA": "Asie",
    "SYDNEY": "Sydney",
    "LONDON": "Londres",
    "NEW_YORK": "New York",
    "LONDON_NY_OVERLAP": "Chevauchement Londres+NY",
    "OFF_SESSION": "Hors session",
}
_DECISION_LABELS = {"proposal": "Proposé", "rejected": "Rejeté"}

_SHOCK_LABELS = {
    "NORMAL": "Normal",
    "VOLATILITY_EXPANSION": "Expansion de volatilité",
    "SHOCK": "CHOC",
}


def _utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _f(value, nd: int = 2, dash: str = "—") -> str:
    if value is None:
        return dash
    return f"{value:.{nd}f}"


# ------------------------------------------------------------ data assembly


def collect(session: Session, limit: int = 5000) -> dict:
    """Gather every dashboard number from the local database."""
    now = datetime.now(timezone.utc)
    today_start = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
    resolved = resolved_signals(session, limit=limit)
    recent = list(
        session.scalars(select(SignalRecord).order_by(SignalRecord.ts.desc()).limit(limit))
    )
    latest = recent[0] if recent else None

    proposals_today = sum(
        1 for r in recent
        if r.final_decision == "proposal" and _utc(r.ts) >= today_start
    )
    rejected_today = sum(
        1 for r in recent
        if r.final_decision == "rejected" and _utc(r.ts) >= today_start
    )
    trades_today = session.scalar(
        select(func.count()).select_from(Position).where(Position.opened_at >= today_start)
    ) or 0

    # ACTIONABLE_SIGNAL_RATE (V-MONSTER §64): proposals over proposals +
    # TOO_LATE aborts — how often the human actually got an actionable
    # signal today. TOO_LATE is the only timing-driven pre-send abort.
    too_late_today = sum(
        1 for r in recent
        if r.final_decision == "rejected"
        and r.no_trade_reason == "TOO_LATE"
        and _utc(r.ts) >= today_start
    )
    actionable_denom = proposals_today + too_late_today
    actionable_rate = (
        round(proposals_today / actionable_denom, 4) if actionable_denom else None
    )

    overall = compute_trade_stats([(r.outcome, r.r_multiple) for r in resolved])
    buckets = confidence_buckets(resolved)

    # Confidence histogram over EVERY proposal (resolved or not).
    conf_hist: Counter = Counter()
    for r in recent:
        if r.final_decision != "proposal":
            continue
        conf = (r.fusion or {}).get("raw_confidence")
        if not isinstance(conf, (int, float)):
            continue
        for label, lo, hi in _CONFIDENCE_BUCKETS:
            if lo <= conf < hi:
                conf_hist[label] += 1
                break

    # Setup-quality histogram + performance per quality tier.
    sq_hist: Counter = Counter()
    for r in recent:
        score = (r.setup_quality or {}).get("score")
        if not isinstance(score, (int, float)):
            continue
        for label, lo, hi in _SQ_HIST_BINS:
            if lo <= score < hi:
                sq_hist[label] += 1
                break
    sq_perf: list[dict] = []
    for label, lo, hi in _SQ_TIERS:
        pairs = [
            (r.outcome, r.r_multiple) for r in resolved
            if isinstance((r.setup_quality or {}).get("score"), (int, float))
            and lo <= (r.setup_quality or {}).get("score") < hi
        ]
        sq_perf.append({"tier": label, "stats": compute_trade_stats(pairs).to_dict()})

    reasons = Counter(
        r.no_trade_reason or "OTHER"
        for r in recent if r.final_decision == "rejected"
    )
    rejection_reasons = [
        {"reason": reason, "count": count} for reason, count in reasons.most_common(15)
    ]

    dims = {}
    for dim, label in _DIM_LABELS.items():
        dims[dim] = {
            "label": label,
            "groups": {k: v.to_dict() for k, v in breakdown(resolved, dim).items()},
        }

    snap = latest.market_snapshot if latest else {}
    fusion = latest.fusion or {} if latest else {}
    news = snap.get("news_context") or []
    next_news = min(
        (e for e in news if (e.get("minutes_to_event") or 0) >= 0),
        key=lambda e: e.get("minutes_to_event") or 0,
        default=None,
    )
    current = {
        "ts": latest.ts.isoformat() if latest else None,
        "symbol": latest.symbol if latest else None,
        "timeframe": latest.timeframe if latest else None,
        "decision": _DECISION_LABELS.get(latest.final_decision, "—") if latest else None,
        "price": snap.get("price"),
        "regime": _REGIME_LABELS.get((snap.get("regime") or {}).get("regime"), "—"),
        "session": _SESSION_LABELS.get((snap.get("session_context") or {}).get("session"), "—"),
        "alignment": (snap.get("alignment") or {}).get("label") or "—",
        "dxy_gauge": snap.get("dxy_gauge") or {},
        "dxy_context": snap.get("dxy_context") or {},
        "confidence": fusion.get("raw_confidence"),
        "calibrated": fusion.get("calibrated_confidence"),
        "setup_quality": (latest.setup_quality or {}).get("score") if latest else None,
        "mtf_biases": snap.get("mtf_biases") or {},
        "shock": (snap.get("shock_context") or {}).get("state"),
        "next_news": next_news,
    }

    recent_signals = []
    for r in recent[:20]:
        fusion = r.fusion or {}
        ds = fusion.get("direction_score")
        side = ("LONG" if (ds or 0) > 0 else "SHORT") if ds is not None else "—"
        recent_signals.append({
            "ts": r.ts.strftime("%Y-%m-%d %H:%M") if r.ts else "—",
            "symbol": r.symbol,
            "decision": _DECISION_LABELS.get(r.final_decision, r.final_decision),
            "side": side,
            "confidence": fusion.get("raw_confidence"),
            "outcome": r.outcome or "—",
            "r": r.r_multiple,
        })

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M UTC"),
        "versions": version_stamp(),
        "current": current,
        "today": {
            "proposals": proposals_today,
            "rejected": rejected_today,
            "trades": trades_today,
            "actionable_rate": actionable_rate,
        },
        "overall": overall.to_dict(),
        "resolved_count": len(resolved),
        "records_analysed": len(recent),
        "confidence_buckets": buckets,
        "confidence_histogram": dict(conf_hist),
        "setup_quality_histogram": dict(sq_hist),
        "setup_quality_performance": sq_perf,
        "rejection_reasons": rejection_reasons,
        "breakdowns": dims,
        "recent_signals": recent_signals,
    }


# ------------------------------------------------------------ HTML rendering

_CSS = """
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #101418;
       color: #dde3ea; margin: 0; padding: 24px 32px 48px; }
h1 { font-size: 22px; margin: 4px 0 2px; }
h2 { font-size: 16px; margin: 28px 0 8px; color: #7fb2ff;
     border-bottom: 1px solid #2a3440; padding-bottom: 4px; }
.sub { color: #8a97a5; font-size: 13px; }
.cards { display: flex; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
.card { background: #1a2129; border: 1px solid #2a3440; border-radius: 8px;
        padding: 12px 18px; min-width: 140px; }
.card .v { font-size: 22px; font-weight: 600; color: #e8eef5; }
.card .k { font-size: 12px; color: #8a97a5; text-transform: uppercase; }
table { border-collapse: collapse; margin-top: 8px; font-size: 13px; }
th, td { border: 1px solid #2a3440; padding: 5px 12px; text-align: right; }
th { background: #1a2129; color: #9fb4c8; font-weight: 600; }
td.l, th.l { text-align: left; }
tr.insufficient td { color: #66707c; }
tr.insufficient td:first-child::after { content: ' *'; color: #b98a2f; }
.bar { display: flex; align-items: center; gap: 8px; margin: 3px 0; }
.bar .label { width: 90px; text-align: right; color: #9fb4c8; font-size: 12px; }
.bar .track { flex: 1; background: #1a2129; border-radius: 4px; height: 16px;
              overflow: hidden; border: 1px solid #2a3440; }
.bar .fill { background: #3d6fb4; height: 100%; }
.bar .n { width: 60px; color: #8a97a5; font-size: 12px; }
.note { color: #66707c; font-size: 12px; margin-top: 10px; }
.grid2 { display: flex; gap: 24px; flex-wrap: wrap; }
.grid2 > div { flex: 1; min-width: 380px; }
.good { color: #6fbf7f; } .bad { color: #d9776f; }
"""


def _esc(value) -> str:
    return html.escape(str(value))


def _stat_table(headers: list[str], rows: list[list], classes=None) -> str:
    head = "".join(f"<th class='l'>{_esc(h)}</th>" if h == headers[0]
                   else f"<th>{_esc(h)}</th>" for h in headers)
    body = []
    for i, row in enumerate(rows):
        cls = f" class='{classes[i]}'" if classes and classes[i] else ""
        cells = []
        for j, cell in enumerate(row):
            left = " class='l'" if j == 0 else ""
            cells.append(f"<td{left}>{_esc(cell)}</td>")
        body.append(f"<tr{cls}>{''.join(cells)}</tr>")
    return f"<table><tr>{head}</tr>{''.join(body)}</table>"


def _bars(histogram: dict) -> str:
    if not histogram:
        return "<p class='sub'>Aucune donnée.</p>"
    peak = max(histogram.values()) or 1
    out = []
    for label, count in histogram.items():
        width = max(2, int(100 * count / peak))
        out.append(
            f"<div class='bar'><span class='label'>{_esc(label)}</span>"
            f"<span class='track'><span class='fill' style='width:{width}%'></span></span>"
            f"<span class='n'>{count}</span></div>"
        )
    return "".join(out)


def render_html(data: dict) -> str:
    """Pure builder: data dict -> self-contained HTML document."""
    c = data["current"]
    overall = data["overall"]
    today = data["today"]

    current_cards = [
        ("Prix", f"{c['price']:,.2f}" if c["price"] else "—"),
        ("Régime", c["regime"]),
        ("Session", c["session"]),
        ("Alignement MTF", c["alignment"]),
        ("Confiance", f"{c['confidence']:.2f}" if c["confidence"] is not None else "—"),
        ("Confiance calibrée", f"{c['calibrated']:.2f}" if c["calibrated"] is not None else "—"),
        ("Qualité du setup", f"{c['setup_quality']:.2f}" if c["setup_quality"] is not None else "—"),
        ("Choc de marché", _SHOCK_LABELS.get(c.get("shock"), "—")),
        (
            "Prochaine news",
            f"{c['next_news']['event']} dans {c['next_news']['minutes_to_event']}m"
            if c.get("next_news") else "—",
        ),
    ]
    dxy_gauge = c["dxy_gauge"]
    dxy_ctx = c["dxy_context"]
    dxy_line = "—"
    if dxy_gauge or dxy_ctx:
        parts = []
        if dxy_gauge:
            parts.append(f"jauge {dxy_gauge.get('value')} ({dxy_gauge.get('classification')})")
        for key in ("direction", "trend", "momentum_pct", "classification"):
            if dxy_ctx.get(key) is not None:
                parts.append(f"{key} {dxy_ctx[key]}")
        dxy_line = " | ".join(parts)
    mtf = c["mtf_biases"]
    mtf_line = " · ".join(f"{tf} {b.get('bias')}" for tf, b in mtf.items()) if mtf else "—"

    cards = "".join(
        f"<div class='card'><div class='v'>{_esc(v)}</div><div class='k'>{_esc(k)}</div></div>"
        for k, v in current_cards
    )

    today_cards = "".join(
        f"<div class='card'><div class='v'>{v}</div><div class='k'>{_esc(k)}</div></div>"
        for k, v in (
            ("Propositions du jour", today["proposals"]),
            ("Signaux rejetés du jour", today["rejected"]),
            (
                "Signaux actionnables",
                f"{today['actionable_rate'] * 100:.1f}%"
                if today["actionable_rate"] is not None else "—",
            ),
            ("Trades ouverts du jour", today["trades"]),
        )
    )

    perf_rows = [
        ["Trades résolus", overall["trades"]],
        ["Taux de réussite", _f(overall["win_rate"])],
        ["Espérance (R)", _f(overall["expectancy_r"])],
        ["R moyen", _f(overall["avg_r"])],
        ["Profit factor", _f(overall["profit_factor"])],
        ["Drawdown max (R)", _f(overall["max_drawdown_r"])],
    ]

    bucket_rows = []
    bucket_classes = []
    for b in data["confidence_buckets"]:
        bucket_rows.append([
            b["bucket"],
            b["trades"],
            b["wins"],
            _f(b["win_rate"]),
            _f(b["expectancy_r"]),
            _f(b["avg_r"]),
        ])
        bucket_classes.append("" if b["sufficient"] else "insufficient")

    sq_perf_rows = [
        [row["tier"], row["stats"]["trades"], _f(row["stats"]["win_rate"]),
         _f(row["stats"]["expectancy_r"]), _f(row["stats"]["avg_r"])]
        for row in data["setup_quality_performance"]
    ]

    rejection_rows = [[r["reason"], r["count"]] for r in data["rejection_reasons"]]

    recent_rows = [
        [r["ts"], r["symbol"], r["decision"], r["side"],
         _f(r["confidence"]) if r["confidence"] is not None else "—",
         r["outcome"], f"{r['r']:+.2f}" if r["r"] is not None else "—"]
        for r in data["recent_signals"]
    ]

    breakdown_html = ""
    for dim, block in data["breakdowns"].items():
        rows = []
        for key, s in sorted(block["groups"].items(), key=lambda kv: -kv[1]["trades"]):
            rows.append([key, s["trades"], _f(s["win_rate"]),
                         _f(s["expectancy_r"]), _f(s["avg_r"]), _f(s["profit_factor"])])
        if rows:
            breakdown_html += (
                f"<h2>Performance par : {_esc(block['label'])}</h2>"
                + _stat_table(
                    ["Groupe", "Trades", "Taux réussite", "Espérance R", "R moyen", "PF"],
                    rows,
                )
            )

    versions = data["versions"]
    version_line = (
        f"stratégie {versions.get('strategy_version', '—')} · "
        f"config {versions.get('config_version', '—')} · "
        f"prompts {versions.get('prompt_version', '—')}"
    )

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<title>Tableau de bord — robot XAUUSD</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Tableau de bord — robot XAUUSD</h1>
<div class="sub">Généré le {_esc(data['generated_at'])} · {_esc(version_line)} ·
{data['resolved_count']} signaux résolus · {data['records_analysed']} cycles analysés</div>

<h2>Dernier cycle d'analyse</h2>
<div class="sub">{_esc(c['symbol'])} {_esc(c['timeframe'])} · {_esc(c['ts'])} ·
décision : {_esc(c['decision'])}</div>
<div class="cards">{cards}</div>
<div class="sub">DXY : {_esc(dxy_line)}</div>
<div class="sub">Biais MTF : {_esc(mtf_line)}</div>

<h2>Aujourd'hui</h2>
<div class="cards">{today_cards}</div>

<h2>Performance globale</h2>
{_stat_table(["Métrique", "Valeur"], perf_rows)}

<h2>Confiance par tranche (résolus)</h2>
{_stat_table(["Tranche", "Trades", "Gains", "Taux réussite", "Espérance R", "R moyen"],
             bucket_rows, bucket_classes)}
<div class="note">* tranche en dessous de l'échantillon minimum : ne pas tirer de
conclusion (§46).</div>

<div class="grid2">
<div>
<h2>Distribution de la confiance (toutes propositions)</h2>
{_bars(data["confidence_histogram"])}
</div>
<div>
<h2>Distribution de la qualité du setup</h2>
{_bars(data["setup_quality_histogram"])}
</div>
</div>

<h2>Performance par qualité du setup</h2>
{_stat_table(["Tranche", "Trades", "Taux réussite", "Espérance R", "R moyen"], sq_perf_rows)}

<h2>Raisons de rejet</h2>
{_stat_table(["Raison", "Nombre"], rejection_rows) if rejection_rows
 else "<p class='sub'>Aucun rejet enregistré.</p>"}

{breakdown_html}

<h2>Derniers signaux</h2>
{_stat_table(["Heure", "Symbole", "Décision", "Sens", "Confiance", "Résultat", "R"],
             recent_rows) if recent_rows else "<p class='sub'>Aucun signal.</p>"}

<div class="note">Rapport construit uniquement depuis la base locale (spec §22/§45/§46)
— aucune donnée externe, aucune exécution automatique. Les décisions restent
humaines.</div>
</body>
</html>"""


def build_dashboard_html(session: Session, limit: int = 5000) -> str:
    """One-call helper for the CLI and tests."""
    return render_html(collect(session, limit))
