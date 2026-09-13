"""CLI entry point for the trading agent platform.

Commands (paper mode):
  init-db                  create the database and seed risk state
  analyze SYMBOL [--tf]    run the full agent pipeline, print a proposal
  approve ID               approve a pending proposal -> opens a paper position
  reject ID [--reason]     reject a pending proposal
  status                   equity, risk state, open positions, pending proposals
  positions                open positions
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

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from trading_agent.agents.base import LLMClient
from trading_agent.agents.orchestrator import Orchestrator, llm_degraded
from trading_agent.config import Settings, get_settings
from trading_agent.data.gold import GoldData, GoldDataError
from trading_agent.data.market import MarketData, MarketDataError
from trading_agent.data.sessions import session_state
from trading_agent.execution.mt5 import MT5Broker, MT5Error
from trading_agent.execution.paper import PaperBroker
from trading_agent.notify.telegram import TelegramNotifier
from trading_agent.risk.engine import RiskEngine
from trading_agent.schema.types import Rejection, SignalProposal
from trading_agent.store import actions
from trading_agent.store.db import init_engine

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
    (24/7) to the gold provider.
    """
    if symbol.upper() in GOLD_SYMBOLS:
        return GoldData(
            settings.exchange_id,
            mt5=mt5,
            dxy_symbol=settings.mt5_dxy_symbol,
            exchange_ids=settings.paxg_exchanges,
        )
    return MarketData(settings.exchange_id)


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
        summary = payload.get("notes", str(payload))[:100]
        if name == "technical":
            headline = f"{payload.get('bias')} (conv {payload.get('conviction')})"
        elif name == "sentiment":
            headline = f"score {payload.get('score')} ({payload.get('tone')})"
        else:
            headline = f"{payload.get('regime')} (strength {payload.get('trend_strength')})"
        t2.add_row(name, verdict.source, verdict.model, f"{headline}: {summary}")
    console.print(t2)


def _print_proposal(proposal: SignalProposal, pending_id: str | None = None) -> None:
    side_style = "green" if proposal.side.value == "long" else "red"
    panel = Panel.fit(
        f"[bold {side_style}]{proposal.side.value.upper()}[/] {proposal.symbol} "
        f"(confidence {proposal.confidence:.2f}, model: {proposal.model})\n"
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
        console.print(f"[yellow]{args.symbol}: no trade — {result.reason}[/]")
        actions.audit("INFO", "proposal_rejected_by_risk",
                      {"symbol": args.symbol, "reason": result.reason})
        return
    proposal_id = actions.save_proposal(result)
    if args.json:
        console.print_json(result.model_dump())
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
        "[bold red]HALTED[/]" if state.halted else "[green]armed[/]",
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
    for col in ("ID", "Symbol", "Side", "Exit Price", "Reason", "PnL", "Closed (UTC)"):
        table.add_column(col)
    for pos in closed:
        table.add_row(
            str(pos.id), pos.symbol, pos.side.upper(),
            f"{pos.exit_price:,.8g}" if pos.exit_price else "-",
            pos.exit_reason or "-",
            f"{pos.pnl:+,.2f}" if pos.pnl is not None else "-",
            pos.closed_at.strftime("%Y-%m-%d %H:%M") if pos.closed_at else "-",
        )
    console.print(table)


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
        f"sessions Londres/NY uniquement ({settings.session_london} / {settings.session_new_york} locales)"
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
                    continue
                proposal_id = actions.save_proposal(result)
                console.print(
                    f"[bold]{symbol}: new {result.side.value.upper()} proposal "
                    f"(confidence {result.confidence:.2f}) — ID {proposal_id}[/]"
                )
                console.print(
                    f"  approve: [green]python -m trading_agent.main approve {proposal_id}[/]"
                )
                if notifier.enabled:
                    notifier.send_signal(result, gauge)
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

    p_positions = sub.add_parser("positions", help="closed positions with PnL")
    p_positions.set_defaults(func=cmd_positions)

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
