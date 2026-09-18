# V-MONSTER — The XAUUSD Decision Engine

> **The AI proposes. The machine disposes. You decide.**

| | |
|---|---|
| **Name** | V-MONSTER (V3) — successor of INTELLIGENCE_V2 |
| **Mission** | Produce a small number of high-quality XAUUSD trade proposals, gated by a deterministic risk engine, executed only by a human |
| **Specification** | The 92-section V-MONSTER spec — **all 12 phases (A–L) delivered** |
| **Verification** | 664 automated tests · 61 test files · walk-forward parity · honest A/B gate |
| **Execution** | Human-in-the-loop. Paper by default; real MT5 orders only with explicit acknowledgment |
| **Deployment** | Railway background worker (24/7 cloud) + Windows/MT5 bridge for broker data |
| **Repository** | github.com/Oussamajm2255/Trade-wali-jomaa |

---

## 1. What This Machine Is

V-MONSTER is an autonomous XAUUSD analysis robot with a human execution
lever. Every 15 minutes during the London, New York and Sydney sessions
it gathers market data, runs three DeepSeek research agents over a
structured snapshot, fuses their verdicts into one number, and pushes
that number through a **deterministic risk engine of ordered hard
gates**. What survives becomes a proposal — sent to your phone with
entry, stop, target, size, an execution zone, a validity deadline and a
grade (A+ / A). Everything else becomes a **recorded rejection with a
written reason**, and reaches Telegram too: a refusal is intelligence
(why *not* to put money, and proof the machine is still analysing).

Three properties make the project what it is:

1. **Depth** — not a signal script. 77 source modules: a canonical
   snapshot layer, twelve deterministic quality engines (liquidity,
   location, room, speed, trigger, stability, timing, tiers,
   supervision, forensics, health, ablation), a forensic memory of
   every decision, and a self-measurement stack that backtests, walks
   forward and A/B-compares the robot against itself.
2. **Determinism** — every number the AI reads is computed locally by
   our code. The AI never calculates math, never invents data, never
   decides anything final. The risk engine cannot be overruled.
3. **Honesty** — the system treats "we don't know" as information, not
   as a bug: no data never blocks, missing axes score a documented
   neutral, proxy data is labelled, and no performance claim is shown
   without a sample size that can support it.

---

## 2. The Five Design Laws

1. **The AI proposes, the machine disposes.** DeepSeek is an analyst.
   The deterministic risk engine is the final authority — on every
   cycle, for every signal, forever.
2. **One coherent snapshot per cycle, fail-closed on data quality.**
   All modules consume the same validated snapshot; a broken data
   cycle stops before any AI call, not after.
3. **Honesty is an engineering feature.** Fabrication is forbidden at
   the architecture level: unknown values are neutral (0.5), "no data"
   never blocks, proxy volume is never presented as gold volume,
   uncalibrated confidence is labelled UNVALIDATED.
4. **The human is the last gate.** The robot never executes alone.
   Proposals wait for `approve`/`reject`; rejections always reach you
   with the full analysis; the robot learns your real reaction time.
5. **The robot studies itself.** Every decision is stored with its
   gate trail; every proposal and rejection is forensically re-sampled
   at 1/3/5/10/30 minutes; backtests replay your reaction latency;
   ablations measure what each feature actually contributes.

---

## 3. Architecture — The Pipeline

```
   EVERY 15m (London / NY / Sydney sessions)            EVERY 60s (24/7)
┌───────────────────────────────────┐        ┌────────────────────────────┐
│ 1. EYES    MT5 → yfinance → PAXG  │        │ position management (SL/TP)│
│    DXY gauge · calendar · quality │        │ supervision of proposals   │
│    → ONE canonical snapshot       │        │ forensics captures         │
└────────────────┬──────────────────┘        │ health score · balance     │
                 ▼                           └────────────────────────────┘
┌───────────────────────────────────┐
│ 2. BRAINS   3 DeepSeek agents     │  technical 45% · regime 35% · DXY 20%
│    structured JSON in, JSON out   │  → fused side + confidence
└────────────────┬──────────────────┘
                 ▼
┌───────────────────────────────────┐
│ 3. QUALITY   12 deterministic     │  setup · location · room · speed ·
│    engines (Phases A–L)           │  trigger · stability · timing ·
│                                   │  tiers · dedup · supervision ...
└────────────────┬──────────────────┘
                 ▼
┌───────────────────────────────────┐
│ 4. JUDGE    risk engine           │  ~20 ordered hard gates, final
│    → PROPOSAL or REJECTION        │  every verdict stored + gate trail
└────────────────┬──────────────────┘
                 ▼
┌───────────────────────────────────┐
│ 5. HUMAN    Telegram message      │  A+/A · execution zone · deadline
│    approve → you trade on MT5     │  supervision follows each proposal
└────────────────┬──────────────────┘
                 ▼
┌───────────────────────────────────┐
│ 6. MEMORY   outcomes · forensics  │  stats · A/B · ablation ·
│    · calibration · rejection audit│  counterfactuals → next decision
└───────────────────────────────────┘
```

### Module map (the real 77 files)

| Layer | Modules | Role |
|---|---|---|
| Data | `data/snapshot.py` (canonical), `market`, `gold`, `quality`, `indicators`, `structure`, `liquidity`, `vwap`, `speed`, `shock`, `regime`, `bias`, `sessions`, `alignment`, `calendar`, `dxy_context`, `gold_context` | Every input, computed deterministically, validated, merged into one snapshot |
| AI | `agents/base`, `technical`, `regime`, `sentiment`, `dxy`, `fallback` | Three DeepSeek research agents + heuristic fallbacks (degraded mode) |
| Fusion | `fusion/engine`, `setup_quality`, `confidence`, `conflicts`, `location`, `room`, `trigger`, `stability`, `timing`, `tier`, `types` | Weighted vote + the twelve quality engines |
| Judge | `risk/engine.py` | Ordered hard gates; the final authority |
| Memory | `store/db`, `models`, `actions`, `opportunity` | SQLite: every signal, gate trail, opportunity lifecycle, outcomes |
| Self-study | `analytics/{stats,compare,walk_forward,sensitivity,monte_carlo,contribution,counterfactual,rejected,ablation,sample_size}`, `backtest/engine`, `dashboard/report` | Statistics, A/B, ablation, forensics math, HTML cockpit |
| Live ops | `ops/health`, `forensics`, `supervision`, `notify/{telegram,dedup}`, `outcome/engine`, `execution/{paper,mt5}` | Health score + kill-switch, post-signal capture, proposal supervision, Telegram, outcomes, execution |
| Core | `main.py` (~1,600 lines), `config.py` | Loop, 18 CLI commands, one centralised config |

---

## 4. Step 1 — The Eyes: Data & The Canonical Snapshot

| Data | Source chain | Notes |
|---|---|---|
| XAUUSD 15m candles | **MT5 broker (your own terminal) first** → yfinance gold futures → PAXG/USD (24/7) | Broker-native prices and spread when connected; `PREFER_MT5_GOLD_DATA=false` forces the public chain |
| XAUUSD 4h / 1h / 1d | same chain | 4h = the directional bias; 1h/1d = snapshot context |
| DXY gauge (0–100) | MT5 `DXY_U6` → `DX-Y.NYB` fallback | 100 = weak dollar = bullish gold |
| Indicators | computed locally (pandas) | EMA 20/50/200, RSI, MACD, ATR, ADX, Bollinger |
| SMC structure | computed locally | BOS, CHoCH, FVGs, order blocks, sweeps, displacement |
| Liquidity map | computed locally | PDH/PDL, PWH/PWL, session ranges, equal highs/lows, swing clusters + quality score |
| VWAP | computed locally | daily + session VWAP, reclaim/rejection, trend — **labelled unavailable on proxy volume** |
| Speed state | computed locally | SLOW / NORMAL / FAST / EXTREME |
| Shock meter | computed locally | NORMAL / volatility expansion / SHOCK (volume ignored on proxy data) |
| Economic calendar | real provider only (Finnhub, opt-in) | HIGH-impact USD events with minutes-to-go; **never invented by the AI** |

Rules that hold the layer together:

- **One snapshot per cycle.** All 77 modules read from the same
  validated snapshot — nobody refetches, nobody computes a private
  variant, no cross-module drift.
- **Fail-closed on quality.** Too few candles, too stale candles, or
  market timestamps ahead of the local clock (clock skew, Phase A)
  stop the cycle before a single AI call — no money spent on garbage.
  Degraded data continues only when explicitly allowed, and is
  labelled on every signal it touches.
- **Timestamp integrity (Phase A).** Every cycle stores
  `data_latency_ms` (fetch duration) and `data_age_s` (age of the
  freshest candle); Telegram send latency is measured per message.
  Freshness is audited, not assumed.
- **Feature toggles (Phase L).** `feature_smc/dxy/vwap/liquidity/
  speed_enabled` remove a group's inputs at the single wiring point —
  the backbone of the ablation harness (§10).

---

## 5. Step 2 — The Brains: Three DeepSeek Agents

**Honest definition first:** DeepSeek does not browse the internet,
read news, or see charts. It receives a structured JSON snapshot of
numbers and reasons over it with its trained pattern knowledge. Three
agents, three strict missions:

| Agent | Weight | Sees | Returns |
|---|---|---|---|
| Technical | 45% | indicators + 4h bias context | `bias`, `conviction` 0–1, support/resistance, reasoning |
| Regime | 35% | ADX, ATR behaviour, EMA alignment | `regime` + `trend_strength` |
| DXY (sentiment) | 20% | gauge + price context | `score` −1…+1, risk tone |

The vote:

```
score = 0.45 × (technical side ±1) × conviction
      + 0.35 × (regime side ±1) × trend_strength
      + 0.20 × sentiment score
score ≥ +0.25 → LONG      score ≤ −0.25 → SHORT      else → NEUTRAL
confidence = |score| (capped at 1.0)
```

**Degradation, honestly.** Every answer must be valid JSON in our
schema. A failed agent falls back to a deterministic local heuristic,
clearly labelled ("degraded mode") on the signal and in the logs.
Two or more failed agents in a row block new proposals entirely
(`agent_failure_block_min`) — a robot that pretends to think while
hallucinating is worse than a robot that admits it. Per-agent
reliability is tracked over a rolling 200-call window, analysis-only:
weights are never modified from it.

**Cost control.** Before any DeepSeek call: kill-switch armed? price
and ATR sane? spread under the configured ceiling? If the market
isn't worth analysing, the three calls are skipped — API balance is
never spent on invalid markets.

---

## 6. The Twelve Deterministic Engines (the depth of the project)

The V-MONSTER upgrade added twelve engines on top of the V2 core.
Each one is pure, deterministic, side-aware, and honest about missing
data (neutral values, never fabricated, never blocking).

| # | Phase | Engine | What it adds | Spec |
|---|---|---|---|---|
| 1 | A | **Timestamp integrity & latency** | per-cycle `data_latency_ms` / `data_age_s`, clock-skew flag, Telegram send latency | §4, §5, §77 |
| 2 | B | **Liquidity map + VWAP** | PDH/PDL, PWH/PWL, session ranges, equal highs/lows, swing clusters, LIQUIDITY_QUALITY 0–1; honest VWAP (unavailable on proxy volume) | §9, §12 |
| 3 | C | **Location quality + room-to-target** | LOCATION_QUALITY 0–1 (proximity, VWAP relation, premium/discount, stop-side shelter); after-cost distance to the opposing pool must leave ≥ 1R, else INSUFFICIENT_ROOM | §28, §29 |
| 4 | D | **Speed + displacement + trigger** | SLOW/NORMAL/FAST/EXTREME from range-per-minute, formation speed and volatility acceleration; DISPLACEMENT_QUALITY; TRIGGER_QUALITY 0–1 kept separate from trigger speed | §10, §27, §30 |
| 5 | E | **Opportunity clustering + dedup** | deterministic OPPORTUNITY_ID; the same opportunity re-signalled within 180m at ±0.1% price is suppressed (SIGNAL_DUPLICATE); pending stronger signal suppresses weaker (OPPORTUNITY_ACTIVE); lifecycle FORMING → TRIGGERED → EXPIRED | §31, §40, §41, §52, §53, §79 |
| 6 | F | **Signal stability + final revalidation** | STABLE/FRAGILE/VERY_FRAGILE from ±1 tick / −1 candle perturbations; one fresh tick must still support the proposal before the send (drift ≤ 0.15%, fresh spread, tick age) | §32, §56 |
| 7 | G | **Timing + actionability** | TIMING_QUALITY 0–1 (maturity, speed, room, drift, lifecycle); SIGNAL_LEAD_TIME; ACTIONABILITY_DEADLINE; EXPECTED_EXECUTION_PRICE over the human reaction window; TOO_LATE aborts the send | §42–§49, §64 |
| 8 | H | **Confidence tiers + grades** | HIGH/MEDIUM/LOW from calibrated win rate per confidence bucket; LOW rejected; MEDIUM capped at 75% size; deterministic A+ / A / NO TRADE labels | §58, §59, §81 |
| 9 | I | **Live supervision + human feedback** | every pending proposal classified each tick: VALID / DO_NOT_CHASE / INVALIDATED / EXPIRED, one Telegram follow-up per state change; your `approve` latency is measured and feeds an EMA that replaces the assumed reaction time | §62, §63 |
| 10 | J | **Forensics + counterfactuals** | every decision (proposal AND rejection) re-sampled at 1/3/5/10/30 min; rejected setups classified CORRECT_REJECT / WRONG_REJECT / INCONCLUSIVE against the realized 30m move; counterfactual entries at T±3/5/10s | §65, §66, §67 |
| 11 | K | **System health + auto-protection** | 0–100 score over six vitals (DB round-trip, Telegram delivery, AI availability, provider quality, tick lag, clock skew); 2 consecutive CRITICAL evaluations engage the kill-switch route; recovery resets only its own halts — a manual halt is sacred | §76, §77, §78 |
| 12 | L | **Realistic backtesting + feature ablation** | backtests replay the trader's real reaction latency with drift-projected fills; WITH vs WITHOUT runs per feature group (SMC / DXY / VWAP / liquidity / speed) with §41-honest verdicts | §69, §70, §72 |

---

## 7. Step 3 — The Judge: Ordered Hard Gates

The risk engine walks the candidate through the gates **in order**.
Any failure = recorded rejection with a written reason and a full
gate trail. The order matters: cheap and global checks first, then
strategy filters, then position math, then the pre-send checks.

**Above everything — the kill-switch.** Daily loss −3%, drawdown
−10%, equity at zero, or a CRITICAL health streak (Phase K): no new
proposals, with a Telegram alert.

| # | Stage | Gate | Rule (defaults) |
|---|---|---|---|
| 0 | pre-AI | Session gate | analysis only during London / New York / Sydney |
| 1 | pre-AI | Data quality | FAIL stops the cycle; DEGRADED labelled |
| 2 | pre-AI | Cost control | spread ceiling exceeded → skip the AI calls |
| 3 | fusion | Neutral | the vote must produce a direction |
| 4 | fusion | Confidence | `confidence ≥ 0.55` |
| 5 | fusion | Conflict detection | ≥ 2 contradictory axes → CONFLICTED → refused |
| 6 | risk | News blackout | opt-in: HIGH-impact USD event within 30 min |
| 7 | risk | Market shock | SHOCK blocks new entries + 30 min cooldown (volume ignored on proxy) |
| 8 | risk | HTF bias | LONG only in 4h uptrend, SHORT only in downtrend; ADX < 20 blocks |
| 9 | risk | DXY concurrency | LONG needs gauge ≥ 55, SHORT needs gauge ≤ 45 |
| 10 | risk | No-trade zones | structure against the trade, consolidation, bad spread, extreme volatility (opt-in) |
| 11 | risk | Room-to-target | after spread + slippage, ≥ 1R to the opposing liquidity pool |
| 12 | risk | Extreme speed | opt-in: EXTREME refuses with ABNORMAL_SPEED |
| 13 | risk | Opportunity dedup | same opportunity re-signal → OPPORTUNITY_ACTIVE / SIGNAL_DUPLICATE |
| 14 | risk | Statistical quality | opt-in: proven negative expectancy on the historical class → refused |
| 15 | risk | Confidence tier | LOW tier (calibrated losing bucket) → refused; MEDIUM capped at 75% size |
| 16 | risk | Duplicates / positions | one open position per symbol; max 3 total |
| 17 | risk | Validity | price and ATR must be sane |
| 18 | risk | Sizing | risk = 1% of equity; size = risk ÷ stop distance; exposure ≤ 25% of equity; dust refused |
| 19 | pre-send | TOO_LATE | expected lead time ≤ human reaction window → abort the send (recorded) |
| 20 | pre-send | Final revalidation | one fresh tick must still support it: drift ≤ 0.15%, spread fresh, tick age ≤ 600 s |

**SL & TP are mechanical:** stop = entry ∓ 2×ATR, target = entry ±
4×ATR — always 2:1, never negotiated.

---

## 8. Step 4 — The Human Loop

A passing signal becomes a **proposal**:

- stored with the **full gate trail** — every gate, pass or fail, with
  its detail (`signal <id>` shows it all);
- graded **A+ / A** (A+ requires: HIGH tier, setup quality ≥ 0.7,
  timing quality ≥ 0.7, room ≥ 2R);
- sent to Telegram with: entry, stop, target, size, grade, the DXY
  line, the 4h bias line, the trigger state, **the execution zone**
  (entry ± 0.5×ATR max chase), the **expected execution price** after
  your reaction time, and the **valid-until deadline**;
- introduced by a short **narrative note** (presentation layer): one
  or two human sentences about what the market is doing and why the
  robot acts — worded never more confidently than the real score, with
  the unvalidated calibration said out loud and the next level to
  watch taken from the stored data only. The full technical trace
  stays underneath (`── Détails (audit) ──`) — nothing is deleted, it
  is reorganised. Disable with `TELEGRAM_NARRATIVE_ENABLED=false`;
- **supervised every tick** (Phase I) until you act:

| Supervision state | Meaning | Robot action |
|---|---|---|
| VALID | price still inside the execution zone, deadline open | waits for you |
| DO_NOT_CHASE | price left the zone (pullback entry still possible) | one follow-up message; stays pending |
| INVALIDATED | stop or target was hit | follow-up; approving is refused |
| EXPIRED | actionability deadline passed | follow-up; approving is refused |

And the loop learns you (Phase G/I): every `approve` measures the
signal-to-execution latency and feeds an EMA (span 10) that replaces
the assumed reaction window in the timing model — the robot prices
its proposals with **your** real reaction time, not a textbook guess.

**Rejections reach you too.** Every refusal is sent with the gate
that fired, the reason, and the agents' full analysis (biases,
reasoning, contributions) — deduplicated into a heartbeat when the
same gate repeats within the hour. A refusal is trader intelligence.

---

## 9. The Self-Measurement System

The robot does not trust itself — it tests itself:

| Tool | Answers | Command |
|---|---|---|
| Statistics engine | win rate, expectancy, calibration per confidence bucket | automatic + dashboard |
| Replay | "what exactly did the robot know at that moment?" | `replay <ts>` |
| Backtest | candle-by-candle historical replay — deterministic AI mode, isolated DB, no look-ahead | `backtest XAUUSD` |
| **Realistic backtest** (Phase L) | the same replay, but the fill is delayed by your reaction window and priced with the drift projection + slippage — opt-in via `BACKTEST_REALISTIC_EXECUTION=true` | `backtest XAUUSD` |
| Walk-forward | "does it hold out-of-sample, or is it curve-fitted?" | `walkforward XAUUSD` |
| Sensitivity | "what if the parameters were slightly different?" | `sensitivity XAUUSD` |
| Monte Carlo | "how bad can drawdown get if trades shuffle?" | `montecarlo XAUUSD` |
| A/B compare | "is the change really better than the baseline?" | `ab-compare XAUUSD` |
| **Feature ablation** (Phase L) | "what does each feature group actually contribute?" WITH vs WITHOUT on the same window, isolated DBs | `analytics/ablation.py` |
| Counterfactual timing (Phase J) | "what if we had entered T±3/5/10s earlier?" | `analytics/counterfactual.py` |
| Rejection audit (Phase J) | "were our rejections right?" — CORRECT_REJECT / WRONG_REJECT / INCONCLUSIVE | dashboard |
| Sample-size floor (Phase L, §36) | "is N large enough to claim anything?" — `required_trades`, `detectable_effect` | `analytics/sample_size.py` |
| Dashboard | one HTML page: performance, confidence buckets, actionability KPI, rejection quality, health | `dashboard` |

**The honesty protocol (§7) governs every claim:**

- **What IMPROVED means** — the A/B verdict needs ≥ `min_trades`
  resolved trades per side, better expectancy, AND drawdown/profit
  factor not meaningfully worse. Fewer trades alone is never an
  improvement. Today, with zero resolved paper trades, the gate
  reports INSUFFICIENT_DATA by design — never a fake win.
- **Sample size (§36)** — no fixed "100 trades" rule. The required N
  for any performance claim is derived from effect size, variance and
  power (`required_trades`), and the inverse (`detectable_effect`) is
  the floor below which every confidence/win-rate number shown to you
  is labelled **UNVALIDATED**.
- **Held-out final validation** — a locked 12-month window that was
  never used for development, tuning or A/B. It is touched exactly
  once, at the very end (one-shot, never re-run to "improve" the
  result): the calibration curve is checked per confidence bucket
  against the §36 sample floors. Verdict: PASS / FAIL /
  INSUFFICIENT_DATA — the UNVALIDATED labels move only on a clear
  PASS. **EXECUTED** on 2025-09-17 → 2026-09-17 (proxy gold chain,
  honestly labelled). Result: **INSUFFICIENT_DATA** — 8 resolved
  trades in 12 months, no bucket anywhere near the §36 floor (393),
  so no calibration claim can be made yet. The UNVALIDATED labels
  stand and the robot stays in paper mode until enough trades
  resolve; the full run record is `heldout_validation_result.json`.

---

## 10. Live Operations

- **Health monitor (Phase K).** The loop scores its own vitals every
  60 s — DB reachability (a real round-trip), Telegram delivery rate
  and latency, AI availability (real answers vs heuristic fallbacks),
  provider data quality, tick lag, clock offset — into one 0–100
  score. ≥ 80 HEALTHY, < 50 CRITICAL, between = DEGRADED. Two
  consecutive CRITICAL evaluations engage the kill-switch route;
  three HEALTHY ones auto-reset **only the halts this monitor
  engaged** — a manual halt is never touched.
- **Railway background worker** — 24/7 in the cloud with the public
  data chain; the PAXG fallback walks Binance → Kraken → OKX to dodge
  geo-blocks automatically.
- **Windows + MT5** — the bridge of choice for real broker data (live
  or paper analysis): real prices, real spread, real tick activity,
  with a 5-minute skip on a dead terminal instead of hammering it.
- **Single production instance** — the design contract forbids two
  robots writing to one account.
- **DeepSeek balance monitoring** — daily check, Telegram alert below
  $1; zero balance = labelled degraded mode, never silent.
- **Every decision is stored** — signals, gate trails, opportunity
  lifecycles, post-signal snapshots, outcomes, audit events. The
  database is the robot's memory, and the memory is queryable.

---

## 11. The Honesty Ledger

What is real, what is proxy, what is open — written down, not
assumed:

| Item | Status | Rule |
|---|---|---|
| Gold candles | real first | MT5 broker feed → yfinance → PAXG proxy fallback, in that order |
| Volume | often proxy | PAXG token flow is **not** gold volume: the shock meter ignores it and VWAP is labelled unavailable on it — never presented as institutional |
| News | real provider only | never invented by the AI; unreachable provider fails open (trades without the filter), opt-in |
| DXY gauge | real with caveat | frozen at Friday close over the weekend (labelled) |
| Confidence calibration | guarded | shown as UNVALIDATED until buckets pass the §36 sample floors |
| No-data behaviour | fail open | missing inputs never block a signal (spec §4); they score a documented neutral |
| Held-out validation | executed — INSUFFICIENT_DATA | one-shot run (2025-09-17 → 2026-09-17): 8 resolved trades, no bucket at the §36 floor — UNVALIDATED stands, paper mode continues |
| Live trading | opt-in | MT5 mode refuses to start without `live_acknowledged=true` |

**Still open by design:** macro, positioning, options, flows and
yields data modules (schema + UNAVAILABLE status only — activate
behind explicit API keys later); volume profile (§13) waits for a
trustworthy volume source; multi-symbol confirmation is on the
roadmap, not in the spec yet.

---

## 12. Worked Simulations

### A — The A+ signal that passes everything

Monday 14:30 UTC, London/NY overlap. Gold 2400, ATR 5.
Agents: technical LONG 0.60 · regime trending-up 0.70 · gauge 62.

```
score = 0.45×0.60 + 0.35×0.70 + 0.20×0.24 = +0.563 → LONG, conf 0.56
setup quality 0.78 (BOS aligned, price above order block, location 0.8)
room to opposing pool 2.6R after costs · speed NORMAL · trigger 0.8
stability STABLE · timing quality 0.74, lead 45 min, deadline 15:15
tier HIGH (calibrated bucket win rate 0.64) → size 1.0 × risk
```

Gates 0–18 pass; revalidation: fresh tick 2399.9 (drift 0.004%) ✓;
lead 45 min > reaction 123 s ✓. **Proposal: BUY @ 2400, SL 2390, TP
2420, grade A+, zone 2397.5–2402.5, valid until 15:15 UTC.** The
supervision loop keeps it VALID while price stays in the zone; you
approve 90 s after the message and the robot records 90 s as your
latency — the next proposal's expected execution price is computed
with that.

### B — Refused by the DXY gate

Same agents say LONG with confidence 0.70, everything aligned — but
the gauge is 48. The dollar is not weak enough for gold longs.

> DXY concurrency: gauge 48 not weak-dollar (need ≥ 55) for a LONG

Stored as a rejection with the full trail, sent to Telegram with the
agents' reasoning. The robot stays flat. That is the gate doing its
job.

### C — Aborted pre-send: TOO_LATE

The pipeline produces a good-looking SHORT at 16:50. Timing quality
is fine (0.62), but the opportunity's expected lead time collapsed to
90 s — shorter than your reaction window (120 s + 3 s delivery).

> TOO_LATE: expected lead 90 s ≤ reaction window 123 s

The send is aborted **before** it reaches you, recorded as a
rejection with a `signal_too_late_pre_send` audit event, and counted
in the ACTIONABLE_SIGNAL_RATE KPI. A signal you cannot act on is not
a signal — the robot refuses to spam you with expired opportunities.

### D — Suppressed duplicate

The same structure event re-triggers 25 minutes later at 2401.2 —
within 180 min and 0.1% of the original signal. The opportunity is
still FORMING/TRIGGERED, so:

> OPPORTUNITY_ACTIVE: a stronger pending proposal already covers this opportunity

No second Telegram message, no second position risk. The opportunity
row is updated; when the TTL (12 h) passes un-triggered it becomes
EXPIRED. One setup, one signal — enforced by machine, not by
discipline.

---

## 13. Specification Coverage

| Phase | Sections | Focus |
|---|---|---|
| A | §4, §5, §77 | timestamp integrity, latency metrics, clock skew |
| B | §9, §12 | liquidity map, honest VWAP |
| C | §28, §29 | location quality, room-to-target |
| D | §10, §27, §30 | market speed, displacement & trigger quality |
| E | §31, §40, §41, §52, §53, §79 | opportunity clustering, dedup, lifecycle |
| F | §32, §56 | signal stability, final revalidation |
| G | §42–§49, §64 | timing, actionability, execution model |
| H | §58, §59, §81 | confidence tiers, A+/A/NO TRADE |
| I | §62, §63 | live supervision, human latency feedback |
| J | §65–§67 | forensics, counterfactuals, rejection audit |
| K | §76–§78 | health score, auto-protection, clock |
| L | §69, §70, §72 | realistic backtesting, feature ablation |

**Verification standing:** 664 tests across 61 files, full suite
green; walk-forward parity checked on every scoring change; the A/B
gate reporting INSUFFICIENT_DATA honestly until real trades exist;
held-out validation data-blocked (see §9).

---

## 14. Today's Configuration (defaults)

- **Entry**: 15m candles · **Bias**: 4h (ADX ≥ 20) · **Sessions**:
  London 08:00–17:00, NY 09:30–17:00, Sydney 07:00–16:00 (local)
- **Fusion**: weights 0.45/0.35/0.20 · side threshold ±0.25 ·
  min confidence 0.55 · setup quality ≥ 0.45
- **On**: DXY gate (55/45) · shock (3×, 30 min) · room gate (≥ 1R) ·
  dedup (180 m / 0.1% / TTL 12 h) · timing gate (reaction 120 s + 3 s,
  max chase 0.5×ATR) · tiers (0.6/0.45, MEDIUM cap 0.75) · A+ floors
  (0.7/0.7/2R) · supervision · forensics (1/3/5/10/30 m) · health
  (80/50, 2×/3×) · rejection alerts · balance alerts
- **Off (opt-in)**: news filter (needs a real provider key) ·
  statistical quality (needs resolved outcomes) · EXTREME-speed block ·
  high-volatility block · realistic backtest execution (A/B baseline
  stays next-open)
- **Execution**: paper mode by default; MT5 orders only with explicit
  live acknowledgment; risk 1% per trade, 2×ATR stop, 2:1 target,
  max 3 positions, 25% exposure cap

---

*This document is the identifier of the V-MONSTER project — the
complete V3 specification (92 sections, 12 phases A–L) is
implemented, tested (664 tests) and shipped. The definitive
engineering contract lives in `V_MONSTER_PLAN.md`; this guide is the
human-readable mirror of what the machine actually does.*
