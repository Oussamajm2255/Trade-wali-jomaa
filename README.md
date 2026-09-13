# AI Trading Agent — Crypto, DeepSeek-powered, human-in-the-loop (paper first)

A production-grade **multi-agent trading automation system** for crypto. Three
DeepSeek-backed analysis agents (technical, regime, sentiment) produce
structured verdicts; a deterministic risk engine (with final authority) turns
them into sized, stopped proposals; **you** approve or reject each one; a paper
broker simulates fills, manages stops/targets, and audits every decision.

> **Risk disclaimer**: this is infrastructure and tooling, not financial
> advice. No system guarantees profits. Automated trading can lose money —
> paper trade until a strategy proves itself over many samples.

## Architecture

```
 Market data (ccxt public API) ──► indicators (deterministic pandas)
                                        │
          ┌─────────────────────────────┼──────────────────────────┐
          ▼                             ▼                          ▼
   Technical agent                Regime agent                Sentiment agent
   (DeepSeek, JSON schema)        (DeepSeek, JSON schema)     (Fear&Greed + LLM)
          └─────────────────────────────┼──────────────────────────┘
                                        ▼
                         Orchestrator fusion (weighted score)
                                        ▼
               Risk engine — deterministic, hard gates, kill-switch
              (position sizing, exposure caps, daily loss, drawdown)
                                        ▼
                        Proposal → YOU approve / reject (CLI)
                                        ▼
              Paper broker (simulated fills, stop/target, PnL)
                                        ▼
                        Audit log — every decision recorded
```

**Design guarantees**

- The LLM only ever returns *validated JSON schemas* — it can never touch
  order execution, sizing, or risk limits.
- Kill-switch (daily loss / max drawdown / equity depletion) persists across
  restarts and blocks new proposals until you review and reset it.
- Degraded mode: without a DeepSeek key, agents fall back to labelled
  deterministic heuristics so the pipeline keeps working.
- Paper mode uses **public market data only** — no private exchange keys.

## Quickstart

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env     # then edit .env and add your DEEPSEEK_API_KEY

python -m trading_agent.main init-db
python -m trading_agent.main analyze BTC/USDT
# -> prints a proposal with an ID, e.g. 3f2c...-abcd

python -m trading_agent.main approve <proposal-id>   # opens a paper position
python -m trading_agent.main reject <proposal-id>    # or reject it

python -m trading_agent.main status                   # equity, positions, pending
python -m trading_agent.main loop --interval 60       # continuous paper trading
python -m trading_agent.main history --limit 30       # audit trail
python -m trading_agent.main reset-halt               # after manual review only
```

## Gold (XAUUSD)

`analyze XAUUSD` (or `loop --symbols XAUUSD`) routes to a dedicated gold
provider: COMEX gold futures (`GC=F`) via yfinance — tracks spot gold
within a small basis — with automatic fallback to `PAXG/USDT` (tokenised
gold, 24/7) when the futures market is closed or Yahoo is unreachable.
Sentiment uses the dollar index (DXY): gold tends to rise when the dollar
weakens. The data source is always shown in `analyze XAUUSD --verbose`.

## Live execution — IC Markets (MT5)

Live mode sends **real orders** to MetaTrader 5. The adapter
(`trading_agent/execution/mt5.py`) is broker-agnostic: IC Markets, FTMO, or
any MT5 broker — only server/login differ. **IC Markets is the recommended
choice**: a direct broker (raw spreads on XAUUSD, no challenge phase),
whereas FTMO is a prop firm whose challenge rules (5% daily / 10% max loss,
consistency rules) add account-termination risk on top of market risk. The
same adapter works for FTMO later — just change `MT5_SERVER`.

**Safety invariants (hard-coded, not configurable)**

- Every order carries server-side SL and TP — no naked positions, ever.
- Orders are idempotent by proposal id (comment tag + magic number) —
  retries/crashes cannot double-open.
- App-owned positions are tracked by broker ticket; unknown positions are
  adopted, never double-managed.
- Kill-switch (`close-all`, or automatic on limit breach in the loop)
  flattens every position on the account.
- Equity is synced from the broker, so daily-loss/drawdown limits run on
  REAL account equity, not local approximations.

**Setup**

```powershell
# 1. Install MetaTrader 5, log into an IC Markets DEMO account,
#    enable Algo Trading (Tools -> Options -> Expert Advisors).
pip install -r requirements-live.txt

# 2. In .env:
#    EXECUTION_MODE=mt5
#    LIVE_ACKNOWLEDGED=true        # hard guard — refuses to run otherwise
#    MT5_LOGIN=<demo account number>
#    MT5_PASSWORD=<demo password>
#    MT5_SERVER=ICMarketsSC-Demo

python -m trading_agent.main loop --interval 60   # live loop
python -m trading_agent.main close-all            # emergency flatten
```

## Configuration (.env)

| Key | Meaning | Default |
|---|---|---|
| `DEEPSEEK_API_KEY` | Your DeepSeek key; empty = heuristic fallback mode | *(empty)* |
| `DEEPSEEK_MODEL` | Exact model ID your account offers | `deepseek-chat` |
| `EXCHANGE_ID` | ccxt exchange for public data (crypto + PAXG fallback) | `binance` |
| `SYMBOLS` | Watchlist (crypto and/or `XAUUSD`) | `BTC/USDT,ETH/USDT,SOL/USDT` |
| `TIMEFRAME` | Candle timeframe | `1h` |
| `PAPER_STARTING_EQUITY` | Simulated account size (USD) | `10000` |
| `RISK_PER_TRADE` | % of equity risked per trade | `0.01` |
| `MAX_POSITIONS` | Concurrent paper positions | `3` |
| `MAX_EXPOSURE` | Portfolio notional cap (% equity) | `0.25` |
| `DAILY_LOSS_LIMIT` | Kill-switch at daily loss % | `0.03` |
| `MAX_DRAWDOWN` | Kill-switch at drawdown from peak % | `0.10` |
| `ATR_STOP_MULT` | Stop distance in ATRs | `2.0` |
| `TAKE_PROFIT_RR` | Reward:risk for targets | `2.0` |
| `MIN_CONFIDENCE` | Minimum fused confidence | `0.55` |
| `EXECUTION_MODE` | `paper` (default) or `mt5` (real orders) | `paper` |
| `LIVE_ACKNOWLEDGED` | Hard guard: must be `true` for mt5 mode | `false` |
| `MT5_LOGIN` / `MT5_PASSWORD` | Broker account (demo first) | *(empty)* |
| `MT5_SERVER` | e.g. `ICMarketsSC-Demo` | `ICMarketsSC-Demo` |
| `MT5_MAGIC` | Orders owned by this app | `770313` |
| `MT5_DEVIATION_POINTS` | Max slippage accepted on entries | `20` |

## Risk model (what the LLM cannot change)

- **Sizing**: `size = (equity × risk_per_trade) / (ATR × ATR_STOP_MULT)`, then
  capped by portfolio exposure and a minimum-notional floor.
- **Stops/targets**: ATR-based, asymmetric `TAKE_PROFIT_RR` targets.
- **Kill-switch**: daily loss limit → halt; drawdown from peak → halt; equity
  depletion → halt. Persistent, audited, manual reset only.
- **Paper fills**: latest closed candle + slippage + taker fee. Exit checks
  are conservative (stop assumed to fill first when both hit in one candle).

## Tests

```powershell
python -m pytest -q
```

## Roadmap (deliberately not in v1)

- Backtest harness (vectorbt) to validate strategies on history
- FastAPI approval dashboard (replaces CLI approvals)
- FTMO profile (same MT5 adapter, prop-firm rules mapped onto the
  kill-switch limits)
- News/sentiment streams beyond Fear & Greed (funding rates, liquidations)
- Postgres + Redis for multi-instance operation
