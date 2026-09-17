"""CLI entry point for the trading agent platform.

Commands (paper mode):
  init-db                  create the database and seed risk state
  analyze SYMBOL [--tf]    run the full agent pipeline, print a proposal
  approve ID               approve a pending proposal -> opens a paper position
  reject ID [--reason]     reject a pending proposal
  status                   equity, risk state, open positions, pending proposals
  positions                closed positions (with outcome metrics)
  signals [--symbol]       recent complete signal records (spec §22)
  signal ID                one full signal record with gate trail
  replay SIGNAL_ID         deterministic replay of what the robot knew (spec §39)
  backtest SYMBOL [--tf]   candle-by-candle historical replay (deterministic)
  walkforward SYMBOL       walk-forward out-of-sample validation (spec §26)
  sensitivity SYMBOL       parameter sensitivity sweeps (spec §27)
  montecarlo SYMBOL        Monte Carlo risk analysis (spec §28)
  quality SYMBOL           regime analytics + conditional expectancy (spec §29/§30)
  ab-compare SYMBOL        A/B run LEGACY_BASELINE vs INTELLIGENCE_V2 (spec §40/§41)
  dashboard [--out]        intelligence dashboard HTML report (spec §45/§46)
  history [--limit N]      recent audit events
  reset-halt               clear the kill-switch after manual review
  loop [--symbols ...]     continuous paper trading loop (stop/target mgmt + proposals)

Every action is written to the audit log.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_agent.agents.base import LLMClient
from trading_agent.agents.orchestrator import Orchestrator, llm_degraded
from trading_agent.backtest.engine import BacktestEngine, BacktestError
from trading_agent.config import Settings, get_settings
from trading_agent.data.calendar import build_calendar_provider
from trading_agent.data.gold import GoldData, GoldDataError
from trading_agent.data.market import MarketData, MarketDataError
from trading_agent.data.sessions import session_state
from trading_agent.execution.mt5 import MT5Broker, MT5Error
from trading_agent.execution.paper import PaperBroker
from trading_agent.notify.dedup import RejectionDedup
from trading_agent.notify.telegram import TelegramNotifier, agent_bias, agent_conviction
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, SignalProposal
from trading_agent.store import actions
from trading_agent.store.db import init_engine, session_scope
from trading_agent.versioning import version_stamp

console = Console()
logger = logging.getLogger("trading_agent")

# Symbols routed to the gold data provider (COMEX futures + PAXG fallback).
GOLD_SYMBOLS = {"XAUUSD", "XAU/USD", "GOLD", "GC=F"}


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
    )
    # Never echo credentials: httpx INFO logs full URLs (bot token included).
    logging.getLogger("httpx").setLevel(logging.WARNING)


def _broker_for(settings: Settings, risk: RiskEngine):
    """Execution backend by EXECUTION_MODE, with hard live-mode guards."""
    if not settings.live_mode:
        return PaperBroker(settings, risk)
    if not settings.live_acknowledged:
        console.print(
            "[bold red]LIVE MODE REFUSED TO START.[/] Set LIVE_ACKNOWLEDGED=true in .env "
            "after reading the broker section of the README."
        )
        sys.exit(1)
    if not settings.mt5_configured:
        console.print("[bold red]MT5 credentials missing.[/] Set MT5_LOGIN, MT5_PASSWORD and MT5_SERVER in .env.")
        sys.exit(1)
    return MT5Broker(settings, risk)


def _components():
    settings = get_settings()
    risk = RiskEngine(settings)
    broker = _broker_for(settings, risk)
    return settings, risk, broker


def _market_for(settings: Settings, symbol: str, mt5=None):
    """Gold symbols use the gold provider; everything else uses crypto ccxt.

    In live mode the connected MT5 broker feeds broker-native DXY candles
    (24/7) to the gold provider. A long-lived economic-calendar provider
    (spec §43) is attached so its 5-minute result cache spans cycles.
    """
    if symbol.upper() in GOLD_SYMBOLS:
        market = GoldData(
            settings.exchange_id,
            mt5=mt5,
            dxy_symbol=settings.mt5_dxy_symbol,
            exchange_ids=settings.paxg_exchanges,
        )
    else:
        market = MarketData(settings.exchange_id)
    market.calendar_provider = build_calendar_provider(settings)  # type: ignore[attr-defined]
    return market


def _print_verdicts(verdicts: dict, snapshot: dict, gauge: dict | None) -> None:
    """Verbose report of what each agent saw and answered."""
    table = Table(title=f"Market snapshot — {snapshot.get('symbol')} {snapshot.get('timeframe')}", box=box.ROUNDED)
    for col in ("Price", "RSI", "ADX", "ATR", "ATR pct", "EMA20>50", "EMA50>200", "24h %", "7d %", "Source"):
        table.add_column(col)
    table.add_row(
        f"{snapshot.get('last_close'):,.2f}" if snapshot.get("last_close") else "-",
        str(snapshot.get("rsi_14", "-")),
        str(snapshot.get("adx_14", "-")),
        f"{snapshot.get('atr_14'):,.2f}" if snapshot.get("atr_14") else "-",
        str(snapshot.get("atr_percentile_100", "-")),
        str(snapshot.get("ema20_gt_ema50", "-")),
        str(snapshot.get("ema50_gt_ema200", "-")),
        str(snapshot.get("return_24h_pct", "-")),
        str(snapshot.get("return_7d_pct", "-")),
        str(snapshot.get("data_source", "-")),
    )
    console.print(table)

    if gauge:
        console.print(
            f"[cyan]Sentiment gauge:[/] {gauge.get('value')} ({gauge.get('classification')}) — "
            f"source: {gauge.get('source', 'n/a')}"
        )

    t2 = Table(title="Agent verdicts", box=box.ROUNDED)
    for col in ("Agent", "Source", "Model", "Verdict"):
        t2.add_column(col)
    for name, verdict in verdicts.items():
        payload = verdict.payload
        summary = (payload.get("reasoning") or payload.get("notes") or str(payload))[:100]
        if name == "technical":
            headline = f"{payload.get('bias')} (conv {payload.get('conviction')})"
        elif name == "sentiment":
            headline = f"score {payload.get('score')} ({payload.get('tone')})"
        elif name == "dxy":
            headline = f"gold {payload.get('gold_bias')} (score {payload.get('score')})"
        else:
            headline = f"{payload.get('regime')} (strength {payload.get('trend_strength')})"
        t2.add_row(name, verdict.source, verdict.model, f"{headline}: {summary}")
    console.print(t2)


def _print_proposal(proposal: SignalProposal, pending_id: str | None = None) -> None:
    side_style = "green" if proposal.side.value == "long" else "red"
    fusion_line = (
        f"dir {proposal.direction_score:+.2f} | raw confidence {proposal.raw_confidence:.2f}"
    )
    if proposal.calibrated_confidence is not None:
        fusion_line += f" | calibrated {proposal.calibrated_confidence:.2f}"
    else:
        fusion_line += " | calibrated n/a (needs outcomes)"
    if proposal.setup_quality:
        sq = proposal.setup_quality
        comps = ", ".join(f"{k} {v:.2f}" for k, v in sq.get("components", {}).items())
        fusion_line += f"\nsetup quality {sq.get('score'):.2f} [{comps}]"
    if proposal.conflicts:
        fusion_line += (
            f"\nconflicts: {proposal.conflicts.get('state')} "
            f"(score {proposal.conflicts.get('conflict_score')})"
        )
    panel = Panel.fit(
        f"[bold {side_style}]{proposal.side.value.upper()}[/] {proposal.symbol} "
        f"(confidence {proposal.confidence:.2f}, model: {proposal.model})\n"
        f"{fusion_line}\n"
        f"entry {proposal.entry:,.8g} | stop {proposal.stop:,.8g} | target {proposal.target:,.8g}\n"
        f"size {proposal.size:,.8g} | risk {proposal.risk_amount:.2f} USD | RR {proposal.expected_rr:.1f}\n\n"
        f"[dim]{proposal.rationale}[/]",
        title=f"Proposal {pending_id or proposal.id}",
        border_style=side_style,
    )
    console.print(panel)
    if pending_id:
        console.print(
            f"Approve: [green]python -m trading_agent.main approve {pending_id}[/]   "
            f"Reject: [red]python -m trading_agent.main reject {pending_id}[/]\n"
        )


def cmd_init_db(_: argparse.Namespace) -> None:
    init_engine()
    console.print("[green]Database initialised.[/]")


def cmd_analyze(args: argparse.Namespace) -> None:
    settings, risk, broker = _components()
    halted, reason = risk.is_halted()
    if halted:
        console.print(f"[bold red]Kill-switch engaged: {reason}[/] — no new analysis. "
                      "Run 'reset-halt' only after manual review.")
        return
    market = _market_for(settings, args.symbol)
    orchestrator = Orchestrator(settings, market, risk)
    try:
        if args.verbose:
            result, verdicts, snapshot, gauge = orchestrator.run_full(args.symbol, args.tf)
            _print_verdicts(verdicts, snapshot, gauge)
        else:
            result = orchestrator.run(args.symbol, args.tf)
    except (MarketDataError, GoldDataError) as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    if isinstance(result, Rejection):
        if result.no_trade_reason:
            console.print(
                f"[yellow]{args.symbol}: no trade ({result.no_trade_reason}) — {result.reason}[/]"
            )
        else:
            console.print(f"[yellow]{args.symbol}: no trade — {result.reason}[/]")
        actions.audit("INFO", "proposal_rejected_by_risk",
                      {"symbol": args.symbol, "reason": result.reason,
                       "no_trade_reason": result.no_trade_reason})
        return
    proposal_id = actions.save_proposal(result)
    actions.link_signal_proposal(result.signal_id, proposal_id)
    if args.json:
        console.print_json(data=result.model_dump())
    _print_proposal(result, pending_id=proposal_id)


def cmd_approve(args: argparse.Namespace) -> None:
    settings, risk, broker = _components()
    halted, reason = risk.is_halted()
    if halted:
        console.print(f"[bold red]Kill-switch engaged: {reason}[/] — cannot approve. "
                      "Run 'reset-halt' only after manual review.")
        sys.exit(1)
    proposal = actions.decide_proposal(args.proposal_id, approve=True, note=args.note or "approved via CLI")
    if proposal is None:
        console.print(f"[red]Proposal {args.proposal_id} not found or not pending.[/]")
        sys.exit(1)
    try:
        position = broker.open_position(proposal)
    except MT5Error as exc:
        actions.revert_proposal(proposal.id, f"execution failed: {exc}")
        console.print(f"[red]Execution failed: {exc}[/] — proposal reverted to pending (retry allowed).")
        sys.exit(1)
    if position is None:
        console.print("[red]Could not open position.[/]")
        sys.exit(1)
    console.print(
        f"[green]Position opened:[/] {proposal.symbol} {proposal.side.value.upper()} "
        f"size {proposal.size:,.8g} @ {position.entry:,.8g} "
        f"(stop {proposal.stop:,.8g}, target {proposal.target:,.8g})"
    )


def cmd_reject(args: argparse.Namespace) -> None:
    _components()
    proposal = actions.decide_proposal(args.proposal_id, approve=False, note=args.reason or "rejected via CLI")
    if proposal is None:
        console.print(f"[red]Proposal {args.proposal_id} not found or not pending.[/]")
        sys.exit(1)
    console.print(f"[yellow]Rejected {proposal.symbol} proposal {proposal.id}.[/]")


def cmd_status(_: argparse.Namespace) -> None:
    _components()
    state = actions.get_risk_state()
    if state is None:
        console.print("[red]No risk state — run init-db first.[/]")
        sys.exit(1)
    day_pnl = state.equity - state.start_of_day_equity
    drawdown = (1 - state.equity / state.peak_equity) * 100 if state.peak_equity else 0.0
    table = Table(title="Paper account", box=box.ROUNDED)
    table.add_column("Equity", justify="right")
    table.add_column("Day PnL", justify="right")
    table.add_column("Peak", justify="right")
    table.add_column("Drawdown", justify="right")
    table.add_column("Positions", justify="right")
    table.add_column("Kill-switch", justify="right")
    table.add_row(
        f"{state.equity:,.2f}",
        f"{day_pnl:+,.2f}",
        f"{state.peak_equity:,.2f}",
        f"{drawdown:.2f}%",
        str(len(actions.open_positions())),
        "[bold red]HALTED[/]" if state.halted else "[green]ready[/]",
    )
    console.print(table)
    if state.halted:
        console.print(f"[bold red]Halt reason: {state.halt_reason}[/]")

    open_pos = actions.open_positions()
    if open_pos:
        t2 = Table(title="Open positions", box=box.ROUNDED)
        for col in ("ID", "Symbol", "Side", "Size", "Entry", "Stop", "Target", "Opened (UTC)"):
            t2.add_column(col)
        for pos in open_pos:
            t2.add_row(
                str(pos.id), pos.symbol, pos.side.upper(), f"{pos.size:,.8g}",
                f"{pos.entry:,.8g}", f"{pos.stop:,.8g}", f"{pos.target:,.8g}",
                pos.opened_at.strftime("%Y-%m-%d %H:%M"),
            )
        console.print(t2)

    pending = actions.list_proposals(status="pending")
    if pending:
        t3 = Table(title="Pending proposals", box=box.ROUNDED)
        for col in ("ID", "Symbol", "Side", "Conf.", "Entry", "Stop", "Target", "Model"):
            t3.add_column(col)
        for prop in pending:
            t3.add_row(
                str(prop.id), prop.symbol, prop.side.value.upper(), f"{prop.confidence:.2f}",
                f"{prop.entry:,.8g}", f"{prop.stop:,.8g}", f"{prop.target:,.8g}", prop.model,
            )
        console.print(t3)
    else:
        console.print("[dim]No pending proposals.[/]")


def cmd_positions(_: argparse.Namespace) -> None:
    _components()
    closed = actions.closed_positions(limit=20)
    table = Table(title="Closed positions", box=box.ROUNDED)
    for col in ("ID", "Symbol", "Side", "Exit", "Reason", "Outcome", "R", "PnL", "Closed (UTC)"):
        table.add_column(col)
    for pos in closed:
        table.add_row(
            str(pos.id), pos.symbol, pos.side.upper(),
            f"{pos.exit_price:,.8g}" if pos.exit_price else "-",
            pos.exit_reason or "-",
            pos.outcome or "-",
            f"{pos.r_multiple:+.2f}" if pos.r_multiple is not None else "-",
            f"{pos.pnl:+,.2f}" if pos.pnl is not None else "-",
            pos.closed_at.strftime("%Y-%m-%d %H:%M") if pos.closed_at else "-",
        )
    console.print(table)


def cmd_signals(args: argparse.Namespace) -> None:
    """Recent complete signal records — proposals AND rejections (§22)."""
    if args.db:
        init_engine(args.db)
    _components()
    rows = actions.list_signals(limit=args.limit, symbol=args.symbol, decision=args.decision)
    if not rows:
        console.print("[dim]No signal records yet — run analyze/loop first.[/]")
        return
    table = Table(title="Signal records", box=box.ROUNDED)
    for col in ("Signal ID", "UTC", "Symbol", "TF", "Decision", "Outcome", "Conf."):
        table.add_column(col)
    for row in rows:
        fusion = row.fusion if isinstance(row.fusion, dict) else {}
        conf = fusion.get("raw_confidence")
        table.add_row(
            row.signal_id,
            row.ts.strftime("%m-%d %H:%M") if row.ts else "-",
            row.symbol,
            row.timeframe,
            row.final_decision,
            row.outcome or "-",
            f"{conf:.2f}" if conf is not None else "-",
        )
    console.print(table)


def cmd_signal(args: argparse.Namespace) -> None:
    """One full signal record: snapshot, fusion, setup quality, gate trail."""
    if args.db:
        init_engine(args.db)
    _components()
    row = actions.get_signal(args.signal_id)
    if row is None:
        console.print(f"[red]Signal {args.signal_id} not found.[/]")
        sys.exit(1)
    fusion = row.fusion if isinstance(row.fusion, dict) else {}
    style = "green" if row.final_decision == "proposal" else "yellow"
    console.print(Panel.fit(
        f"[bold]{row.signal_id}[/] {row.symbol} {row.timeframe} — "
        f"[bold {style}]{row.final_decision}[/]"
        + (f" (outcome {row.outcome}, R {row.r_multiple:+.2f})" if row.outcome else "")
        + f"\nstrategy {row.strategy_version} | config {row.config_version} | "
        f"prompts {row.prompt_version}\n"
        + (f"confidence {fusion.get('raw_confidence')}" if fusion.get("raw_confidence") is not None else "")
        + (f" | direction {fusion.get('direction_score'):+.2f}" if fusion.get("direction_score") is not None else "")
        + (f" | calibrated {fusion.get('calibrated_confidence')}" if fusion.get("calibrated_confidence") is not None else "")
        + "\n"
        + (f"SL {row.sl:,.8g} | TP {row.tp:,.8g} | size {row.size:,.8g} | risk {row.risk_amount:.2f} USD\n" if row.final_decision == "proposal" else "")
        + f"reason: {row.decision_reason or '-'}\n"
        + f"no-trade class: {row.no_trade_reason or '-'}",
        title="Signal record",
        border_style=style,
    ))
    gates = row.gates if isinstance(row.gates, list) else []
    if gates:
        t = Table(title="Gate trail", box=box.ROUNDED)
        for col in ("Gate", "Status", "Detail"):
            t.add_column(col)
        for g in gates:
            t.add_row(
                g.get("gate", "-"), g.get("status", "-"), str(g.get("detail", ""))[:70]
            )
        console.print(t)
    snap = row.market_snapshot if isinstance(row.market_snapshot, dict) else {}
    if snap.get("price"):
        console.print(
            f"[dim]price {snap['price']:,.8g} | regime {snap.get('regime', {}).get('regime', 'n/a')} "
            f"| quality {snap.get('data_quality', 'n/a')} | source {snap.get('data_source', 'n/a')}[/]"
        )
    else:
        console.print(f"[dim]snapshot: {snap.get('error', 'unavailable')}[/]")


def cmd_replay(args: argparse.Namespace) -> None:
    """Deterministic signal replay (spec §39): reconstruct exactly what
    the robot knew for one signal — read from the stored record, never
    recomputed, with statistical context stopping at decision time."""
    from trading_agent.analytics.contribution import feature_contribution
    from trading_agent.analytics.stats import compute_trade_stats, resolved_signals

    if args.db:
        init_engine(args.db)
    _components()
    row = actions.get_signal(args.signal_id)
    if row is None:
        console.print(f"[red]Signal {args.signal_id} not found.[/]")
        sys.exit(1)
    record = row.to_dict()
    snap = record.get("market_snapshot") or {}

    # Statistical context at decision time: only signals resolved BEFORE
    # this one (no look-ahead) — the replay is fully deterministic.
    with session_scope() as session:
        hist = resolved_signals(session, limit=args.context, before=row.ts, symbol=row.symbol)
    stats = compute_trade_stats([(r.outcome, r.r_multiple) for r in hist])
    contribution = feature_contribution(record)

    payload = {
        "signal_id": row.signal_id,
        "ts": row.ts.isoformat() if row.ts else None,
        "symbol": row.symbol,
        "timeframe": row.timeframe,
        "strategy_version": row.strategy_version,
        "decision": row.final_decision,
        "reason": row.decision_reason,
        "no_trade_reason": row.no_trade_reason,
        "market_snapshot": snap,
        "ai_outputs": record.get("ai_outputs") or {},
        "fusion": record.get("fusion") or {},
        "setup_quality": record.get("setup_quality"),
        "conflicts": record.get("conflicts"),
        "gates": record.get("gates") or [],
        "sl": row.sl,
        "tp": row.tp,
        "size": row.size,
        "risk_amount": row.risk_amount,
        "outcome": row.outcome,
        "r_multiple": row.r_multiple,
        "contribution": contribution.to_dict(),
        "statistical_context": {"resolved_before": len(hist), "stats": stats.to_dict()},
    }
    if args.json:
        console.print_json(data=payload)
        return

    style = "green" if row.final_decision == "proposal" else "yellow"
    fusion = record.get("fusion") or {}
    header = (
        f"[bold]{row.signal_id}[/] {row.symbol} {row.timeframe} — "
        f"[bold {style}]{row.final_decision}[/]"
        + (f" (outcome {row.outcome}, R {row.r_multiple:+.2f})" if row.outcome else "")
        + f"\nstrategy {row.strategy_version} | at {row.ts}"
        + (f"\nconfidence {fusion.get('raw_confidence')}" if fusion.get("raw_confidence") is not None else "")
        + (f" | direction {fusion.get('direction_score'):+.2f}" if fusion.get("direction_score") is not None else "")
        + (f" | calibrated {fusion.get('calibrated_confidence')}" if fusion.get("calibrated_confidence") is not None else "")
        + (f"\nSL {row.sl:,.8g} | TP {row.tp:,.8g} | size {row.size:,.8g} | risk {row.risk_amount:.2f} USD" if row.final_decision == "proposal" else "")
        + f"\nreason: {row.decision_reason or '-'} (no-trade {row.no_trade_reason or '-'})"
    )
    console.print(Panel.fit(header, title="Signal replay (spec §39)", border_style=style))

    ctx = []
    if snap.get("price"):
        ctx.append(f"price {snap['price']:,.8g}")
    regime = (snap.get("regime") or {}).get("regime")
    if regime:
        ctx.append(f"regime {regime}")
    session = (snap.get("session_context") or {}).get("session")
    if session:
        ctx.append(f"session {session}")
    if snap.get("data_quality"):
        ctx.append(f"quality {snap['data_quality']}")
    if snap.get("data_source"):
        ctx.append(f"source {snap['data_source']}")
    if snap.get("last_close"):
        ctx.append(f"close {snap['last_close']:,.8g}")
    for key in ("rsi_14", "adx_14", "atr_14"):
        if snap.get(key) is not None:
            ctx.append(f"{key} {snap[key]}")
    console.print("[bold]Market snapshot[/] " + " | ".join(ctx) if ctx else "[dim]snapshot unavailable[/]")

    dxy_gauge = snap.get("dxy_gauge") or {}
    dxy_ctx = snap.get("dxy_context") or {}
    dxy_parts = []
    if dxy_gauge:
        dxy_parts.append(f"gauge {dxy_gauge.get('value')} ({dxy_gauge.get('classification')})")
    if dxy_ctx:
        for key in ("direction", "trend", "momentum_pct", "classification"):
            if dxy_ctx.get(key) is not None:
                dxy_parts.append(f"{key} {dxy_ctx[key]}")
    console.print("[bold]DXY[/] " + " | ".join(dxy_parts) if dxy_parts else "[dim]DXY n/a[/]")

    mtf = snap.get("mtf_biases") or {}
    if mtf:
        console.print("[bold]Timeframes[/] " + " · ".join(
            f"{tf}: {b.get('bias')}" for tf, b in mtf.items()
        ))
    structure = snap.get("structure") or {}
    present = [k for k in ("bos", "choch", "fvgs", "sweeps") if structure.get(k)]
    if present:
        console.print(f"[bold]Structure[/] {', '.join(present)}")

    quality = record.get("setup_quality") or {}
    conflicts = record.get("conflicts") or {}
    console.print(
        f"[bold]Quality[/] score {quality.get('score', 'n/a')} | "
        f"components {quality.get('components') or '-'} | conflict state {conflicts.get('state', 'n/a')}"
    )

    ai = record.get("ai_outputs") or {}
    if ai:
        t = Table(title="AI outputs (stored)", box=box.ROUNDED)
        for col in ("Agent", "Bias", "Conviction", "Source"):
            t.add_column(col)
        for name, verdict in sorted(ai.items()):
            verdict = verdict or {}
            payload = verdict.get("payload") or {}
            conviction = payload.get("conviction")
            t.add_row(
                name,
                str(payload.get("bias") or payload.get("side") or "-"),
                f"{conviction:.2f}" if isinstance(conviction, (int, float)) else "-",
                verdict.get("source", "-"),
            )
        console.print(t)

    gates = record.get("gates") or []
    if gates:
        t = Table(title="Risk gates (stored)", box=box.ROUNDED)
        for col in ("Gate", "Status", "Detail"):
            t.add_column(col)
        for g in gates:
            t.add_row(g.get("gate", "-"), g.get("status", "-"), str(g.get("detail", ""))[:70])
        console.print(t)

    console.print(
        f"[bold]Statistical context at decision time[/] {len(hist)} resolved before | "
        f"win rate {_fmt_stat(stats.win_rate)} | expectancy {_fmt_stat(stats.expectancy_r)} R | "
        f"PF {_fmt_stat(stats.profit_factor)}"
    )

    if contribution.supporting:
        console.print(f"[green]Soutient :[/] " + " · ".join(contribution.supporting))
    if contribution.contradicting:
        console.print(f"[red]Contredit :[/] " + " · ".join(contribution.contradicting))
    if contribution.invalidation:
        console.print(f"[yellow]Invalidation :[/] {contribution.invalidation}")


def _fetch_frames(
    settings: Settings, symbol: str, tf: str, limit: int
) -> tuple[dict, object | None]:
    """Fetch multi-timeframe history (and DXY) for offline analytics.

    Entry TF is mandatory; other timeframes degrade like live.
    """
    market = _market_for(settings, symbol)
    tfs = [tf] + [t for t in settings.snapshot_timeframes if t != tf]
    frames: dict = {}
    for t in tfs:
        try:
            frames[t] = market.fetch_ohlcv(symbol, t, limit)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - one missing TF degrades like live
            logger.warning("analytics frame %s unavailable: %s", t, exc)
    dxy = None
    if hasattr(market, "dxy_ohlcv"):
        try:
            dxy = market.dxy_ohlcv("1h", limit)  # type: ignore[attr-defined]
        except Exception as exc:  # noqa: BLE001 - gauge/context degrade like live
            logger.warning("analytics DXY history unavailable: %s", exc)
    return frames, dxy


def cmd_backtest(args: argparse.Namespace) -> None:
    """Candle-by-candle historical replay (spec §24) — deterministic AI."""
    settings = get_settings()
    symbol = args.symbol
    tf = args.tf or settings.timeframe
    limit = args.limit or settings.backtest_history_limit
    frames, dxy = _fetch_frames(settings, symbol, tf, limit)
    if tf not in frames:
        console.print(f"[red]Entry timeframe {tf} data unavailable — aborting.[/]")
        sys.exit(1)
    engine = BacktestEngine(
        settings=settings,
        frames=frames,
        symbol=symbol,
        timeframe=tf,
        dxy_frames=dxy,
        start=args.start,
        end=args.end,
        db_url=args.db,
        spread_pct=args.spread,
        warmup=args.warmup,
    )
    try:
        report = engine.run()
    except BacktestError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    _print_backtest(report, args.json)


def _print_backtest(report, as_json: bool = False) -> None:
    if as_json:
        console.print_json(data=report.to_dict())
        return
    stats = report.stats
    console.print(Panel.fit(
        f"[bold]{report.symbol} {report.timeframe}[/] {report.start_ts} -> {report.end_ts} "
        f"({report.candles} candles)\n"
        f"trades {stats['trades']} | wins {stats['wins']} | losses {stats['losses']} | "
        f"win rate {stats['win_rate']}\n"
        f"profit factor {stats['profit_factor']} | expectancy {stats['expectancy_r']} R | "
        f"total PnL {stats['total_pnl']:+,.2f} USD\n"
        f"max drawdown {stats['max_drawdown_pct']}% | final equity {stats['final_equity']:,.2f}",
        title=f"Backtest — deterministic AI (DB: {report.db_url})",
    ))
    for halt in report.halt_events:
        console.print(f"[bold red]KILL-SWITCH at {halt['ts']}: {halt['reason']}[/]")
    if report.trades:
        table = Table(title="Trades", box=box.ROUNDED)
        for col in ("Signal", "Side", "Entry", "Exit", "Reason", "Outcome", "R", "PnL", "Bars"):
            table.add_column(col)
        for t in report.trades:
            table.add_row(
                (t.signal_id or "-")[-14:],
                t.side.upper(),
                f"{t.entry:,.8g}" if t.entry else "-",
                f"{t.exit_price:,.8g}" if t.exit_price else "-",
                t.exit_reason or "-",
                t.outcome or "-",
                f"{t.r_multiple:+.2f}" if t.r_multiple is not None else "-",
                f"{t.pnl:+,.2f}" if t.pnl is not None else "-",
                str(t.bars_open),
            )
        console.print(table)


def _fmt_stat(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def cmd_walkforward(args: argparse.Namespace) -> None:
    """Walk-forward validation (spec §26): consecutive out-of-sample replays."""
    from trading_agent.analytics.walk_forward import WalkForwardConfig, run_walk_forward

    settings = get_settings()
    symbol = args.symbol
    tf = args.tf or settings.timeframe
    limit = args.limit or settings.backtest_history_limit
    frames, dxy = _fetch_frames(settings, symbol, tf, limit)
    if tf not in frames:
        console.print(f"[red]Entry timeframe {tf} data unavailable — aborting.[/]")
        sys.exit(1)
    config = WalkForwardConfig(
        train_bars=args.train, val_bars=args.val, test_bars=args.test, step_bars=args.step
    )
    try:
        report = run_walk_forward(
            settings, frames, config, symbol=symbol, timeframe=tf,
            dxy_frames=dxy, warmup=args.warmup,
        )
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    if args.json:
        console.print_json(data=report.to_dict())
        return
    agg = report.aggregate
    console.print(Panel.fit(
        f"[bold]{report.symbol} {report.timeframe}[/] walk-forward "
        f"train {report.config.train_bars} / val {report.config.val_bars} / "
        f"test {report.config.test_bars} bars (step {report.config.step_bars})\n"
        f"windows {agg['windows']} | trades {agg['total_trades']} | "
        f"profitable windows {agg['profitable_windows']}/{agg['windows']} "
        f"({agg['profitable_ratio']})\n"
        f"avg expectancy {agg['avg_expectancy_r']} R (median {agg['median_expectancy_r']}, "
        f"stdev {agg['stdev_expectancy_r']}) | avg win rate {agg['avg_win_rate']}",
        title="Walk-forward validation (out-of-sample)",
    ))
    table = Table(title="Windows", box=box.ROUNDED)
    for col in ("Test window", "Candles", "Trades", "Win rate", "Expectancy R", "Total R"):
        table.add_column(col, justify="right")
    for w in report.windows:
        s = w.stats
        table.add_row(
            f"{w.test_start[11:16]} -> {w.test_end[11:16]} {w.test_start[:10]}",
            str(w.candles), str(s.get("trades", 0)), _fmt_stat(s.get("win_rate")),
            _fmt_stat(s.get("expectancy_r")), _fmt_stat(s.get("total_r")),
        )
    console.print(table)


def cmd_sensitivity(args: argparse.Namespace) -> None:
    """Parameter sensitivity (spec §27): one deterministic replay per grid value."""
    from trading_agent.analytics.sensitivity import ALLOWED_PARAMETERS, run_sensitivity

    settings = get_settings()
    symbol = args.symbol
    tf = args.tf or settings.timeframe
    limit = args.limit or settings.backtest_history_limit
    frames, dxy = _fetch_frames(settings, symbol, tf, limit)
    if tf not in frames:
        console.print(f"[red]Entry timeframe {tf} data unavailable — aborting.[/]")
        sys.exit(1)
    grid: dict[str, list[float]] = {}
    for spec in args.param:
        name, _, raw = spec.partition("=")
        name = name.strip()
        if not name or not raw:
            console.print(f"[red]--param must be name=v1,v2,... (got '{spec}')[/]")
            sys.exit(1)
        if name not in ALLOWED_PARAMETERS:
            console.print(
                f"[red]unknown parameter '{name}' — allowed: {sorted(ALLOWED_PARAMETERS)}[/]"
            )
            sys.exit(1)
        grid[name] = [float(v) for v in raw.split(",") if v.strip()]
    if not grid:
        console.print("[yellow]No --param given: nothing to sweep.[/]")
        sys.exit(1)
    report = run_sensitivity(
        settings, frames, grid, symbol=symbol, timeframe=tf,
        dxy_frames=dxy, warmup=args.warmup, spread_pct=args.spread,
    )
    if args.json:
        console.print_json(data=report.to_dict())
        return
    table = Table(title=f"Sensitivity — {report.symbol} {report.timeframe}", box=box.ROUNDED)
    for col in ("Parameter", "Value", "Trades", "Win rate", "Expectancy R", "PF", "Max DD %"):
        table.add_column(col, justify="right")
    for p in report.points:
        s = p.stats
        table.add_row(
            p.parameter, f"{p.value:g}", str(s.get("trades", 0)), _fmt_stat(s.get("win_rate")),
            _fmt_stat(s.get("expectancy_r")), _fmt_stat(s.get("profit_factor")),
            _fmt_stat(s.get("max_drawdown_pct")),
        )
    console.print(table)
    regions = report.stable_regions("expectancy_r")
    best = report.best_stable("expectancy_r")
    lines = []
    for parameter, spans in regions.items():
        label = ", ".join(f"{r['start']:g}..{r['end']:g}" for r in spans) or "none"
        lines.append(f"{parameter}: stable ranges [{label}] | best stable {best.get(parameter)}")
    console.print(Panel.fit(
        "\n".join(lines),
        title="Stability (prefer stable regions over peak results — spec §27)",
    ))


def cmd_montecarlo(args: argparse.Namespace) -> None:
    """Monte Carlo risk analysis (spec §28) over a deterministic backtest."""
    from trading_agent.analytics.monte_carlo import run_monte_carlo

    settings = get_settings()
    symbol = args.symbol
    tf = args.tf or settings.timeframe
    limit = args.limit or settings.backtest_history_limit
    frames, dxy = _fetch_frames(settings, symbol, tf, limit)
    if tf not in frames:
        console.print(f"[red]Entry timeframe {tf} data unavailable — aborting.[/]")
        sys.exit(1)
    engine = BacktestEngine(
        settings=settings, frames=frames, symbol=symbol, timeframe=tf,
        dxy_frames=dxy, db_url="sqlite:///:memory:",
        spread_pct=args.spread, warmup=args.warmup,
    )
    try:
        run = engine.run()
    except BacktestError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    trade_rs = [t.r_multiple for t in run.trades if t.r_multiple is not None]
    if not trade_rs:
        console.print("[yellow]No trades in the backtest — Monte Carlo needs a sample.[/]")
        sys.exit(1)
    report = run_monte_carlo(
        trade_rs, n=args.n, seed=args.seed,
        risk_per_trade=args.risk, ruin_pct=args.ruin,
    )
    if args.json:
        payload = report.to_dict()
        payload["sample_trades"] = len(trade_rs)
        console.print_json(data=payload)
        return
    console.print(Panel.fit(
        f"[bold]{report.n_simulations} randomized orderings[/] of "
        f"{len(trade_rs)} trade R multiples (seed {report.seed})\n"
        f"risk of ruin (equity <= {args.ruin * 100:.0f}% start): "
        f"{report.risk_of_ruin * 100:.1f}%\n"
        f"max drawdown pct    p5 {report.max_drawdown_pct['p5']} | "
        f"p50 {report.max_drawdown_pct['p50']} | p95 {report.max_drawdown_pct['p95']}\n"
        f"max losing streak   p5 {report.max_losing_streak['p5']:.0f} | "
        f"p50 {report.max_losing_streak['p50']:.0f} | p95 {report.max_losing_streak['p95']:.0f}\n"
        f"final equity        p5 {report.final_equity['p5']:,.0f} | "
        f"p50 {report.final_equity['p50']:,.0f} | p95 {report.final_equity['p95']:,.0f}",
        title="Monte Carlo risk analysis — NOT a profitability claim (spec §28)",
    ))


def cmd_quality(args: argparse.Namespace) -> None:
    """Regime analytics / conditional expectancy (spec §29/§30)."""
    from trading_agent.analytics.stats import (
        breakdown, compute_trade_stats, conditional_expectancy, resolved_signals,
    )

    if args.db:
        init_engine(args.db)
    settings = get_settings()
    symbol = args.symbol
    with session_scope() as session:
        rows = resolved_signals(session, limit=args.limit, symbol=symbol)
    if not rows:
        console.print("[dim]No resolved signals yet — run the loop and close positions first.[/]")
        return
    conditions: dict = {}
    for spec in args.condition:
        key, _, value = spec.partition("=")
        if key and value:
            conditions[key.strip()] = value.strip()
    if args.json:
        payload: dict = {
            "symbol": symbol,
            "rows": len(rows),
            "overall": compute_trade_stats(
                [(r.outcome, r.r_multiple) for r in rows]
            ).to_dict(),
            "breakdown": {
                dim: {k: v.to_dict() for k, v in breakdown(rows, dim).items()}
                for dim in args.dimension
            },
        }
        if conditions:
            payload["conditional"] = conditional_expectancy(rows, conditions).to_dict()
        console.print_json(data=payload)
        return
    if conditions:
        stats = conditional_expectancy(rows, conditions)
        console.print(Panel.fit(
            f"[bold]{symbol}[/] conditional expectancy where {conditions}\n"
            f"trades {stats.trades} | wins {stats.wins} | win rate {_fmt_stat(stats.win_rate)} | "
            f"expectancy {_fmt_stat(stats.expectancy_r)} R | PF {_fmt_stat(stats.profit_factor)} | "
            f"max DD {_fmt_stat(stats.max_drawdown_r)} R",
            title="Conditional expectancy (spec §30)",
        ))
        return
    for dim in args.dimension or ["side", "regime", "session"]:
        groups = breakdown(rows, dim)
        table = Table(title=f"Performance by {dim} (spec §29) — {symbol}", box=box.ROUNDED)
        for col in ("Group", "Trades", "Wins", "Win rate", "Expectancy R", "PF", "Max DD R"):
            table.add_column(col, justify="right")
        for key in sorted(groups, key=lambda k: -groups[k].trades):
            s = groups[key]
            table.add_row(
                key, str(s.trades), str(s.wins), _fmt_stat(s.win_rate),
                _fmt_stat(s.expectancy_r), _fmt_stat(s.profit_factor),
                _fmt_stat(s.max_drawdown_r),
            )
        console.print(table)
    console.print(
        f"[dim]min sample for statistical quality: {settings.min_sample_for_statistics} "
        f"(gate enabled: {settings.statistical_quality_enabled})[/]"
    )


def cmd_ab_compare(args: argparse.Namespace) -> None:
    """A/B comparison LEGACY_BASELINE vs INTELLIGENCE_V2 (spec §40/§41)."""
    from trading_agent.analytics.compare import run_comparison

    settings = get_settings()
    symbol = args.symbol
    tf = args.tf or settings.timeframe
    limit = args.limit or settings.backtest_history_limit
    frames, dxy = _fetch_frames(settings, symbol, tf, limit)
    if tf not in frames:
        console.print(f"[red]Entry timeframe {tf} data unavailable — aborting.[/]")
        sys.exit(1)
    try:
        report = run_comparison(
            settings, frames, symbol=symbol, timeframe=tf, dxy_frames=dxy,
            start=args.start, end=args.end, db_url=args.db,
            spread_pct=args.spread, warmup=args.warmup, min_trades=args.min_trades,
        )
    except BacktestError as exc:
        console.print(f"[red]{exc}[/]")
        sys.exit(1)
    if args.json:
        console.print_json(data=report.to_dict())
        return
    a = report.results["legacy_baseline"]
    b = report.results["intelligence_v2"]
    table = Table(
        title=f"A/B comparison — {symbol} {tf} ({report.start_ts} -> {report.end_ts})",
        box=box.ROUNDED,
    )
    for col in ("Metric", "LEGACY_BASELINE", "INTELLIGENCE_V2"):
        table.add_column(col)
    for metric, key, fmt in (
        ("Trades", "trades", str),
        ("Win rate", "win_rate", _fmt_stat),
        ("Expectancy R", "expectancy_r", _fmt_stat),
        ("Profit factor", "profit_factor", _fmt_stat),
        ("Max drawdown %", "max_drawdown_pct", _fmt_stat),
        ("Total PnL USD", "total_pnl", lambda v: f"{v:+,.2f}"),
        ("Halt events", "halt_events", str),
    ):
        table.add_row(metric, fmt(a.report.stats.get(key)), fmt(b.report.stats.get(key)))
    console.print(table)
    style = {
        "IMPROVED": "green", "MIXED": "yellow", "WORSE": "red", "INSUFFICIENT_DATA": "yellow",
    }.get(b.verdict, "white")
    console.print(
        f"[bold {style}]Verdict : INTELLIGENCE_V2 {b.verdict}[/] — {b.verdict_note}\n"
        f"[dim]Générer moins de trades n'est jamais une amélioration en soi (§41).[/]\n"
        f"[dim]baseline DB {a.report.db_url} | v2 DB {b.report.db_url}[/]"
    )


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Intelligence dashboard (spec §45): self-contained HTML report."""
    from trading_agent.dashboard.report import build_dashboard_html

    if args.db:
        init_engine(args.db)
    _components()
    with session_scope() as session:
        page = build_dashboard_html(session)
    out = Path(args.out) if args.out else Path("dashboard.html")
    out.write_text(page, encoding="utf-8")
    console.print(f"[green]Dashboard écrit dans {out.resolve()}[/]")
    if args.serve:
        _serve_dashboard(page, args.port)


def _serve_dashboard(page: str, port: int) -> None:
    """Serve the static report on 127.0.0.1 (local only, no deps)."""
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    body = page.encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 - http.server API
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_: object) -> None:  # quiet server
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    console.print(
        f"[green]Dashboard: http://127.0.0.1:{port}[/] (Ctrl+C pour arrêter)"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("\n[dim]Serveur arrêté.[/]")


def cmd_agent_stats(args: argparse.Namespace) -> None:
    """Per-agent reliability stats (spec §15) — analysis-only."""
    _components()
    stats = actions.agent_reliability(agent=args.agent, limit=args.limit)
    agents = stats["agents"]
    if not agents:
        console.print("[dim]No agent history yet — run the loop/analyze first.[/]")
        return
    table = Table(title=f"AI reliability (rolling {stats['window']} rows — analysis only)", box=box.ROUNDED)
    for col in ("Agent", "Total", "LLM", "Fallback", "Failures", "Evaluated", "Accuracy", "Avg conf"):
        table.add_column(col, justify="right")
    for agent, s in sorted(agents.items()):
        accuracy = f"{s['accuracy'] * 100:.1f}%" if s["accuracy"] is not None else "-"
        avg_conf = f"{s['avg_confidence']:.2f}" if s["avg_confidence"] is not None else "-"
        table.add_row(
            agent, str(s["total"]), str(s["llm"]), str(s["fallback"]), str(s["failures"]),
            str(s["evaluated"]), accuracy, avg_conf,
        )
    console.print(table)
    console.print(
        "[dim]Accuracy is empty until the outcome engine (phase 5) fills actual outcomes. "
        "Weights are never modified from this data.[/]"
    )


def cmd_history(args: argparse.Namespace) -> None:
    _components()
    rows = actions.recent_audit(limit=args.limit)
    table = Table(title="Audit trail", box=box.ROUNDED)
    for col in ("Time (UTC)", "Level", "Event", "Detail"):
        table.add_column(col)
    for row in rows:
        detail = json.dumps(row.detail)[:80] if row.detail else ""
        table.add_row(
            row.ts.strftime("%Y-%m-%d %H:%M:%S"), row.level, row.event, detail
        )
    console.print(table)


def cmd_reset_halt(_: argparse.Namespace) -> None:
    _, risk, _ = _components()
    risk.reset_halt()
    console.print("[yellow]Kill-switch reset. This is audited.[/]")


def cmd_close_all(_: argparse.Namespace) -> None:
    """Kill-switch on demand: flatten every open position."""
    settings, risk, broker = _components()
    if settings.live_mode:
        try:
            broker.connect()
            closed = broker.close_all()
        except MT5Error as exc:
            console.print(f"[red]{exc}[/]")
            sys.exit(1)
        console.print(f"[bold yellow]KILL EXECUTED: {closed} broker position(s) closed.[/]")
        return
    prices: dict[str, float] = {}
    for pos in actions.open_positions():
        try:
            market = _market_for(settings, pos.symbol)
            df = market.fetch_ohlcv(pos.symbol, settings.timeframe, 50)
            prices[pos.symbol] = float(df.iloc[-1]["close"])
        except Exception as exc:  # noqa: BLE001 - one bad feed must not block others
            logger.warning("no price for %s: %s", pos.symbol, exc)
    events = broker.close_all(prices)
    for event in events:
        style = "green" if event["pnl"] >= 0 else "red"
        console.print(
            f"[{style}]CLOSED {event['symbol']}: {event['exit_reason']} "
            f"@ {event['exit_price']:,.8g} — PnL {event['pnl']:+,.2f} USD[/]"
        )
    if not events:
        console.print("[dim]No open positions to close.[/]")


def cmd_signal_test(_: argparse.Namespace) -> None:
    """Send a test notification — proves the phone is in the loop."""
    settings = get_settings()
    notifier = TelegramNotifier(settings)
    if not notifier.enabled:
        console.print(
            "[yellow]Telegram non configuré.[/] Dans .env : TELEGRAM_BOT_TOKEN et TELEGRAM_CHAT_ID\n"
            "Chat ID : python -m trading_agent.main telegram-chatid"
        )
        sys.exit(1)
    ok = notifier.send_test()
    console.print(
        "[green]Signal de test envoyé sur votre téléphone — vous êtes dans la boucle.[/]"
        if ok else "[red]Échec d'envoi — vérifiez le token et le chat id.[/]"
    )


def cmd_telegram_chatid(args: argparse.Namespace) -> None:
    """Print chat ids that have messaged the bot (so the user can fill .env)."""
    settings = get_settings()
    token = args.token or settings.telegram_bot_token
    if not token:
        console.print("[red]Aucun token — passez --token ou mettez TELEGRAM_BOT_TOKEN dans .env.[/]")
        sys.exit(1)
    try:
        import httpx

        resp = httpx.get(f"https://api.telegram.org/bot{token}/getUpdates", timeout=10)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]getUpdates failed: {exc}[/]")
        sys.exit(1)
    chat_ids = sorted({
        u["message"]["chat"]["id"]
        for u in resp.json().get("result", [])
        if "message" in u
    })
    if chat_ids:
        console.print("Chat IDs trouvés : " + ", ".join(str(c) for c in chat_ids))
        console.print(f"Mettez dans .env : [green]TELEGRAM_CHAT_ID={chat_ids[0]}[/]")
    else:
        console.print(
            "[yellow]Aucun message reçu par le bot.[/] "
            "Ouvrez Telegram, trouvez votre bot, envoyez-lui /start, puis relancez cette commande."
        )


def cmd_loop(args: argparse.Namespace) -> None:
    settings, risk, broker = _components()
    symbols = args.symbols or settings.symbol_list
    timeframe = settings.timeframe
    markets: dict[str, object] = {}
    orchestrators: dict[str, Orchestrator] = {}
    seen: dict[str, object] = {}
    # Rejection phone alerts are deduplicated per symbol: the first
    # refusal of a gate code sends immediately, repeats only re-send as
    # an availability heartbeat (telegram_rejection_repeat_minutes).
    reject_dedup = {
        s: RejectionDedup(settings.telegram_rejection_repeat_minutes) for s in symbols
    }
    halt_banner_shown = False
    kill_executed = False
    halt_notified = False
    sleep_banner_shown = False
    mode_label = "LIVE (MT5)" if settings.live_mode else "Paper"
    notifier = TelegramNotifier(settings)
    balance_client = LLMClient(settings)
    last_balance_day: str | None = None
    last_degraded_day: str | None = None
    mt5_source = broker if settings.live_mode else None
    if settings.live_mode:
        try:
            broker.connect()
        except MT5Error as exc:
            console.print(f"[red]{exc}[/]")
            sys.exit(1)
    session_note = (
        f"sessions Londres/NY/Sydney ({settings.session_london} / {settings.session_new_york} / {settings.session_sydney} locales)"
        if settings.session_filter_enabled
        else "toutes heures"
    )
    console.print(
        f"[bold]{mode_label} loop started[/] — symbols: {', '.join(symbols)}, "
        f"timeframe {timeframe}, tick {args.interval}s, analyse {session_note}. Ctrl+C to stop."
    )
    in_loop = f"Vous êtes dans la boucle : {', '.join(symbols)} ({timeframe}), mode {mode_label}."
    console.print(f"[bold green]✅ {in_loop}[/]")
    if notifier.enabled:
        notifier.send_startup(symbols, timeframe, mode_label)
    else:
        console.print(
            "[yellow]Telegram non configuré[/] — les signaux s'affichent ici uniquement. "
            "Configurez TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID dans .env pour les recevoir sur le téléphone."
        )
    while True:
        try:
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            # Daily DeepSeek balance check: alert before the balance hits
            # zero and every signal silently degrades to heuristics.
            if notifier.enabled and last_balance_day != today:
                last_balance_day = today
                balance = balance_client.balance_usd() if balance_client.enabled else None
                if balance is not None and balance < settings.deepseek_balance_warn_usd:
                    msg = (
                        f"Solde DeepSeek faible : {balance:.2f} USD (seuil "
                        f"{settings.deepseek_balance_warn_usd:.2f}). Rechargez sur "
                        "platform.deepseek.com — sans solde, le robot bascule en mode dégradé."
                    )
                    console.print(f"[bold yellow]{msg}[/]")
                    notifier.send_alert(msg)

            for symbol in symbols:
                market = markets.setdefault(symbol, _market_for(settings, symbol, mt5_source))
                orchestrator = orchestrators.setdefault(
                    symbol, Orchestrator(settings, market, risk)  # type: ignore[arg-type]
                )
                df = market.fetch_ohlcv(symbol, timeframe, settings.ohlcv_limit)  # type: ignore[attr-defined]
                candle = df.iloc[-1]

                for event in broker.manage(symbol, candle):
                    style = "green" if event["pnl"] >= 0 else "red"
                    console.print(
                        f"[{style}]CLOSED {event['symbol']}: {event['exit_reason']} "
                        f"@ {event['exit_price']:,.8g} — PnL {event['pnl']:+,.2f} USD[/]"
                    )

                halted, reason = risk.is_halted()
                if halted:
                    if not halt_banner_shown:
                        console.print(Panel.fit(
                            f"[bold red]KILL-SWITCH ENGAGED: {reason}[/]\n"
                            "No new proposals will be generated.\n"
                            "Review, then run: python -m trading_agent.main reset-halt"
                        ))
                        halt_banner_shown = True
                    if settings.live_mode and not kill_executed:
                        try:
                            closed = broker.close_all()
                        except MT5Error as exc:
                            logger.error("kill close_all failed: %s", exc)
                        else:
                            console.print(f"[bold red]KILL EXECUTED: {closed} position(s) closed on the broker.[/]")
                            kill_executed = True
                    if notifier.enabled and not halt_notified:
                        notifier.send_alert(f"KILL-SWITCH: {reason}")
                        halt_notified = True
                    continue

                # Session gate: analysis sleeps outside London/NY, but
                # position management above stays active 24/7.
                if settings.session_filter_enabled:
                    session = session_state(
                        london=settings.session_london,
                        new_york=settings.session_new_york,
                        sydney=settings.session_sydney,
                    )
                    if not session["in_session"]:
                        if not sleep_banner_shown:
                            console.print(
                                f"[dim]🌙 Hors session — analyse en veille, gestion des positions active. "
                                f"Prochaine ouverture : {session['next_open'][11:16]} UTC.[/]"
                            )
                            sleep_banner_shown = True
                        continue
                    if sleep_banner_shown:
                        console.print(
                            f"[green]Session ouverte[/] — analyse active"
                            + (" (chevauchement Londres+NY 💎)" if session["overlap"] else "") + "."
                        )
                        sleep_banner_shown = False

                last_ts = df.index[-1]
                if seen.get(symbol) == last_ts:
                    continue  # no new closed candle yet
                seen[symbol] = last_ts
                result, verdicts, snapshot, gauge = orchestrator.run_full(symbol, timeframe)

                # Degraded-mode alert: all agents fell back to heuristics
                # (LLM unreachable, e.g. empty balance). Once per UTC day.
                if (not verdicts or llm_degraded(verdicts)) and last_degraded_day != today:
                    last_degraded_day = today
                    msg = (
                        "Mode dégradé : DeepSeek injoignable (solde vide ?) — les signaux "
                        "utilisent les heuristiques locales, moins fines. Rechargez sur "
                        "platform.deepseek.com."
                    )
                    console.print(f"[bold yellow]{msg}[/]")
                    if notifier.enabled:
                        notifier.send_alert(msg)

                if isinstance(result, Rejection):
                    console.print(f"[dim]{symbol}: no trade — {result.reason}[/]")
                    # Compact agent summary so the console log alone
                    # explains the refusal; Telegram carries the full
                    # analysis (biases, reasoning, contributions, gates).
                    agent_pieces = []
                    for name, v in sorted(verdicts.items()):
                        payload = (v.payload or {}) if v else {}
                        piece = f"{name} {agent_bias(payload) or '?'}"
                        conv = agent_conviction(payload)
                        if conv is not None:
                            piece += f" {conv:.2f}"
                        agent_pieces.append(piece)
                    if agent_pieces:
                        console.print(f"[dim]  agents: {' · '.join(agent_pieces)}[/]")
                    # Every rejected opportunity stores its data-quality state
                    # and strategy version (spec §4/§22 — audit the NO-trades).
                    actions.audit(
                        "INFO",
                        "proposal_rejected_by_risk",
                        {
                            "symbol": symbol,
                            "reason": result.reason,
                            "data_quality": snapshot.get("data_quality", "n/a"),
                            "strategy_version": version_stamp()["strategy_version"],
                        },
                    )
                    # Phone alert for refused opportunities (§38): the
                    # refusal and its reason are trader intelligence.
                    # Consecutive refusals by the SAME gate are not
                    # spammed — dedup sends the first, then a heartbeat
                    # once per telegram_rejection_repeat_minutes.
                    if notifier.enabled and result.signal_id:
                        row = actions.get_signal(result.signal_id)
                        if row is not None:
                            code = result.no_trade_reason or result.reason
                            send, note = reject_dedup[symbol].should_send(code)
                            if send:
                                notifier.send_rejection(row.to_dict(), note=note)
                    continue
                proposal_id = actions.save_proposal(result)
                actions.link_signal_proposal(result.signal_id, proposal_id)
                console.print(
                    f"[bold]{symbol}: new {result.side.value.upper()} proposal "
                    f"(confidence {result.confidence:.2f}) — ID {proposal_id}[/]"
                )
                console.print(
                    f"  approve: [green]python -m trading_agent.main approve {proposal_id}[/]"
                )
                if notifier.enabled:
                    row = actions.get_signal(result.signal_id) if result.signal_id else None
                    notifier.send_signal(result, gauge, row.to_dict() if row else None, proposal_id)
            time.sleep(args.interval)
        except KeyboardInterrupt:
            console.print("\n[dim]Loop stopped.[/]")
            if settings.live_mode:
                broker.shutdown()
            return
        except MarketDataError as exc:
            console.print(f"[yellow]Market data error: {exc} — retrying...[/]")
            time.sleep(max(args.interval, 30))
        except Exception as exc:  # noqa: BLE001 - loop must survive anything
            logger.exception("unexpected error in loop iteration: %s", exc)
            time.sleep(max(args.interval, 30))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trading_agent", description="AI trading agent (paper mode, human-in-the-loop)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-db", help="create database and seed risk state")
    p_init.set_defaults(func=cmd_init_db)

    p_analyze = sub.add_parser("analyze", help="run the agent pipeline for one symbol")
    p_analyze.add_argument("symbol")
    p_analyze.add_argument("--tf", default=None, help="timeframe override, e.g. 15m/1h/4h")
    p_analyze.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_analyze.add_argument("--verbose", action="store_true", help="show market snapshot + agent verdicts")
    p_analyze.set_defaults(func=cmd_analyze)

    p_approve = sub.add_parser("approve", help="approve a proposal and open a paper position")
    p_approve.add_argument("proposal_id")
    p_approve.add_argument("--note", default="", help="decision note for the audit log")
    p_approve.set_defaults(func=cmd_approve)

    p_reject = sub.add_parser("reject", help="reject a pending proposal")
    p_reject.add_argument("proposal_id")
    p_reject.add_argument("--reason", default="", help="reason for the audit log")
    p_reject.set_defaults(func=cmd_reject)

    p_status = sub.add_parser("status", help="account, positions, pending proposals")
    p_status.set_defaults(func=cmd_status)

    p_positions = sub.add_parser("positions", help="closed positions with outcome metrics")
    p_positions.set_defaults(func=cmd_positions)

    p_signals = sub.add_parser("signals", help="recent complete signal records (spec §22)")
    p_signals.add_argument("--symbol", default=None, help="filter by symbol")
    p_signals.add_argument("--decision", default=None, choices=["proposal", "rejected"], help="filter by final decision")
    p_signals.add_argument("--limit", type=int, default=50)
    p_signals.add_argument("--db", default=None, help="database override (e.g. a backtest DB)")
    p_signals.set_defaults(func=cmd_signals)

    p_signal = sub.add_parser("signal", help="one full signal record with gate trail")
    p_signal.add_argument("signal_id")
    p_signal.add_argument("--db", default=None, help="database override (e.g. a backtest DB)")
    p_signal.set_defaults(func=cmd_signal)

    p_replay = sub.add_parser("replay", help="deterministic replay of what the robot knew (spec §39)")
    p_replay.add_argument("signal_id")
    p_replay.add_argument("--context", type=int, default=200, help="resolved signals read before this one")
    p_replay.add_argument("--db", default=None, help="database override (e.g. a backtest DB)")
    p_replay.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_replay.set_defaults(func=cmd_replay)

    p_backtest = sub.add_parser("backtest", help="candle-by-candle historical replay (deterministic)")
    p_backtest.add_argument("symbol")
    p_backtest.add_argument("--tf", default=None, help="timeframe override, e.g. 15m/1h/4h")
    p_backtest.add_argument("--start", default=None, help="ISO start (default: earliest data)")
    p_backtest.add_argument("--end", default=None, help="ISO end (default: latest data)")
    p_backtest.add_argument("--limit", type=int, default=None, help="candles fetched per timeframe")
    p_backtest.add_argument("--db", default=None, help="isolated backtest DB (default: BACKTEST_DB_URL)")
    p_backtest.add_argument("--spread", type=float, default=None, help="round-trip spread pct (default: BACKTEST_SPREAD_PCT)")
    p_backtest.add_argument("--warmup", type=int, default=None, help="warmup candles (default: OHLCV_LIMIT)")
    p_backtest.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_backtest.set_defaults(func=cmd_backtest)

    p_wf = sub.add_parser("walkforward", help="walk-forward out-of-sample validation (spec §26)")
    p_wf.add_argument("symbol")
    p_wf.add_argument("--tf", default=None, help="timeframe override, e.g. 15m/1h/4h")
    p_wf.add_argument("--limit", type=int, default=None, help="candles fetched per timeframe")
    p_wf.add_argument("--train", type=int, default=1000, help="train candles per window (must cover warmup)")
    p_wf.add_argument("--val", type=int, default=0, help="validation candles per window")
    p_wf.add_argument("--test", type=int, default=500, help="out-of-sample test candles per window")
    p_wf.add_argument("--step", type=int, default=None, help="window step in candles (default: --test)")
    p_wf.add_argument("--warmup", type=int, default=None, help="indicator warmup candles (default: OHLCV_LIMIT)")
    p_wf.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_wf.set_defaults(func=cmd_walkforward)

    p_sens = sub.add_parser("sensitivity", help="parameter sensitivity sweeps (spec §27)")
    p_sens.add_argument("symbol")
    p_sens.add_argument("--tf", default=None, help="timeframe override, e.g. 15m/1h/4h")
    p_sens.add_argument("--limit", type=int, default=None, help="candles fetched per timeframe")
    p_sens.add_argument(
        "--param", action="append", default=[],
        help="parameter=value1,value2,... (repeatable). Allowed: min_confidence, "
        "setup_quality_min, dxy_long_min, dxy_short_max, atr_stop_mult, take_profit_rr",
    )
    p_sens.add_argument("--warmup", type=int, default=None, help="indicator warmup candles")
    p_sens.add_argument("--spread", type=float, default=None, help="round-trip spread pct")
    p_sens.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_sens.set_defaults(func=cmd_sensitivity)

    p_mc = sub.add_parser("montecarlo", help="Monte Carlo risk analysis over a backtest (spec §28)")
    p_mc.add_argument("symbol")
    p_mc.add_argument("--tf", default=None, help="timeframe override, e.g. 15m/1h/4h")
    p_mc.add_argument("--limit", type=int, default=None, help="candles fetched per timeframe")
    p_mc.add_argument("--n", type=int, default=2000, help="simulations (default 2000)")
    p_mc.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    p_mc.add_argument("--risk", type=float, default=0.01, help="risk per trade as equity fraction")
    p_mc.add_argument("--ruin", type=float, default=0.5, help="ruin threshold: equity fraction of start")
    p_mc.add_argument("--warmup", type=int, default=None, help="indicator warmup candles")
    p_mc.add_argument("--spread", type=float, default=None, help="round-trip spread pct")
    p_mc.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_mc.set_defaults(func=cmd_montecarlo)

    p_qual = sub.add_parser("quality", help="regime analytics + conditional expectancy (spec §29/§30)")
    p_qual.add_argument("symbol")
    p_qual.add_argument(
        "--dimension", action="append", default=[],
        help="breakdown dimension (side/regime/session/alignment/structure/dxy), repeatable",
    )
    p_qual.add_argument(
        "--condition", action="append", default=[],
        help="filter condition key=value, e.g. regime=TREND_UP (repeatable)",
    )
    p_qual.add_argument("--limit", type=int, default=5000, help="resolved signals read")
    p_qual.add_argument("--db", default=None, help="database override (e.g. a backtest DB)")
    p_qual.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_qual.set_defaults(func=cmd_quality)

    p_ab = sub.add_parser("ab-compare", help="A/B comparison LEGACY_BASELINE vs INTELLIGENCE_V2 (spec §40/§41)")
    p_ab.add_argument("symbol")
    p_ab.add_argument("--tf", default=None, help="timeframe override, e.g. 15m/1h/4h")
    p_ab.add_argument("--start", default=None, help="ISO start (default: earliest data)")
    p_ab.add_argument("--end", default=None, help="ISO end (default: latest data)")
    p_ab.add_argument("--limit", type=int, default=None, help="candles fetched per timeframe")
    p_ab.add_argument("--db", default=None, help="base DB URL (per-strategy siblings derived)")
    p_ab.add_argument("--spread", type=float, default=None, help="round-trip spread pct")
    p_ab.add_argument("--warmup", type=int, default=None, help="indicator warmup candles")
    p_ab.add_argument("--min-trades", type=int, default=10, help="minimum resolved trades per side for a verdict")
    p_ab.add_argument("--json", action="store_true", help="print machine-readable JSON")
    p_ab.set_defaults(func=cmd_ab_compare)

    p_dash = sub.add_parser("dashboard", help="generate the intelligence dashboard HTML (spec §45/§46)")
    p_dash.add_argument("--db", default=None, help="database override (e.g. a backtest DB)")
    p_dash.add_argument("--out", default=None, help="output HTML path (default: dashboard.html)")
    p_dash.add_argument("--serve", action="store_true", help="serve on 127.0.0.1 (local only)")
    p_dash.add_argument("--port", type=int, default=8000, help="port for --serve")
    p_dash.set_defaults(func=cmd_dashboard)

    p_stats = sub.add_parser("agent-stats", help="per-agent AI reliability stats (analysis only)")
    p_stats.add_argument("--agent", default=None, help="filter to one agent")
    p_stats.add_argument("--limit", type=int, default=200, help="rolling window size")
    p_stats.set_defaults(func=cmd_agent_stats)

    p_history = sub.add_parser("history", help="recent audit events")
    p_history.add_argument("--limit", type=int, default=30)
    p_history.set_defaults(func=cmd_history)

    p_reset = sub.add_parser("reset-halt", help="clear the kill-switch (audited)")
    p_reset.set_defaults(func=cmd_reset_halt)

    p_close = sub.add_parser("close-all", help="kill-switch on demand: close every open position")
    p_close.set_defaults(func=cmd_close_all)

    p_sigtest = sub.add_parser("signal-test", help="send a test notification to your phone")
    p_sigtest.set_defaults(func=cmd_signal_test)

    p_chatid = sub.add_parser("telegram-chatid", help="print the chat ids that messaged your bot")
    p_chatid.add_argument("--token", default="", help="bot token (default: .env)")
    p_chatid.set_defaults(func=cmd_telegram_chatid)

    p_loop = sub.add_parser("loop", help="continuous paper trading loop")
    p_loop.add_argument("--symbols", nargs="*", default=None, help="symbols to watch (default: .env)")
    p_loop.add_argument("--interval", type=int, default=60, help="tick seconds (default 60)")
    p_loop.set_defaults(func=cmd_loop)

    return parser


def main() -> None:
    _setup_logging()
    # Windows consoles default to a legacy codepage; force UTF-8 output
    # so em-dashes/box glyphs render instead of replacement characters.
    if sys.stdout and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
