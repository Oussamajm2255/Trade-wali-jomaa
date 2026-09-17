# V-MONSTER — Audit, Gap Analysis & Implementation Plan (V3)

Spec: V-MONSTER (92 sections). Base: INTELLIGENCE_V2 (57-section spec,
fully delivered, 394 tests green, running in paper mode on Railway).

Rule 1 of the spec is AUDIT BEFORE MODIFYING. This document is that audit
plus the implementation plan (§88/§92). Nothing here deletes working V2
behaviour; every phase reuses V2 modules and is delivered incrementally
with tests, commit and push gates.

---

## 1. Current V2 architecture (module map)

| Area | Modules |
|---|---|
| Market data | `data/market.py`, `data/gold.py` (proxy chain), `data/snapshot.py` (one coherent snapshot/cycle), `data/quality.py` (good/degraded/fail, fail-closed) |
| Structure/SMC | `data/structure.py` (swings, BOS/CHoCH, FVG, order blocks, displacement candles, sweeps, equal highs), `data/alignment.py` (MTF alignment) |
| Context | `data/gold_context.py` (PDH/PDL, session high/low, DXY distance), `data/dxy_context.py` (gauge 0-100), `data/bias.py`, `data/indicators.py` (EMA/RSI/ATR/ADX), `data/regime.py`, `data/calendar.py` (news blackout/caution, opt-in Finnhub), `data/sessions.py` (London/NY/Sydney gates + labels), `data/shock.py` (shock detection + cooldown, proxy-aware volume) |
| AI agents | `agents/technical.py`, `agents/regime.py`, `agents/sentiment.py`, `agents/dxy.py` (+ `agents/fallback.py` heuristics, degraded-mode safe), `agents/orchestrator.py` (pipeline + gate trail + signal recording) |
| Fusion | `fusion/engine.py`, `fusion/setup_quality.py` (7 weighted components), `fusion/conflicts.py`, `fusion/confidence.py` (calibration) |
| Risk | `risk/engine.py` (deterministic, kill-switch, daily loss, max DD, hard no-trade gates, position sizing) |
| Execution | `execution/paper.py`, `execution/mt5.py` (live mode, human-in-the-loop approve/reject) |
| Analytics | `analytics/stats.py`, `analytics/compare.py` (A/B), `analytics/walk_forward.py`, `analytics/monte_carlo.py`, `analytics/sensitivity.py`, `analytics/contribution.py`, `backtest/engine.py` |
| Storage | `store/db.py`, `store/models.py` (full signal records incl. rejections), `store/actions.py` (audit, proposals, outcomes), `versioning.py` (strategy/config/prompt stamps) |
| Notify | `notify/telegram.py` (full proposal + rejection-with-analysis formats), `notify/dedup.py` (rejection dedup heartbeat) |
| Ops | `main.py` loop (tick 60s, analysis on new 15m candle, sessions, balance alerts, degraded alerts), `dashboard/report.py` |

---

## 2. Gap mapping — the 92 sections vs V2

**EXISTS (do not rebuild):** §6 MTF (15m/1h/4h/1d), §7 structure core,
§8 SMC core, §11 indicators, §23 news (opt-in), §34 conflict (2-state),
§35/§36/§37 statistics + sample windows + calibration, §38 contribution,
§50 hard gates (shock/spread/stale/news/HTF/statistical), §51 shock +
cooldown, §54 AI-as-analyst + deterministic risk, §55 decision order,
§57 final sizing after gates, §60 deterministic risk engine, §67 rejected
setups stored, §69 walk-forward/Monte Carlo/sensitivity/A-B, §82 no-trade
intelligence, §85 DB/logging, §86 versioning, §87 failure-safe, §89 tests.

**PARTIAL (extend):**
- §4 Data quality — no latency/data-age/clock-drift tracking; no provider disagreement.
- §9 Liquidity map — only PDH/PDL, session high/low, equal highs. Missing: weekly/monthly levels, Asia/London/NY session highs, liquidity target/taken/remaining, liquidity quality.
- §10 Displacement — one threshold (range > 1.5×ATR). Missing DISPLACEMENT_QUALITY, body/range ratio, speed, follow-through.
- §22 Session intelligence — classification exists; no session-statistics learning.
- §24 Correlation — DXY divergence only; no yields/equities/VIX axes.
- §25 Regime — trend/range/volatility; no compression/breakout/transition state machine.
- §26 Compression/expansion — volatility state exists; no FORMING/READY/TRIGGERING lifecycle.
- §28 Location quality — distance-to-PDH exists; no 0-1 LOCATION_QUALITY.
- §29 Room-to-target — risk_reward component only; no explicit room vs opposing liquidity.
- §30 Trigger — structure events used; no TRIGGER_QUALITY vs TRIGGER_SPEED separation.
- §33 Quality decomposition — 7 components; spec dimensions (liquidity, location, trigger, timing, stability…) missing.
- §39 Adaptive threshold — static min_confidence only.
- §52/§53 Dedup — rejection-level only; no proposal dedup, no opportunity suppression.
- §61 Telegram — full V2 format; missing setup type, opportunity id, execution zone, max chase, valid-until, actionability deadline, timing/execution/stability scores.
- §63 Human execution feedback — MFE/MAE stored; no execution-latency learning.
- §65/§66 Forensics — MFE/MAE on close; no 1s..30m snapshots, no counterfactual timing.
- §67 Rejected setups — stored with reason; expected-vs-actual outcome not tracked.
- §71/§72/§76/§78/§79 — regime breakdown partial; no human-delay in backtest; no health/clock monitoring; no auto kill on latency; no full decision-state machine.
- §83/§84 KPIs/dashboard — rich but missing actionable-rate, latency KPIs.

**MISSING (new modules):**
- §12 VWAP (session/daily/weekly/anchored), §13 volume profile
  (blocked on reliable volume — see §3).
- §14-§21 macro/real-yields/Fed/risk-regime/positioning/OI/options/flows —
  data-blocked on Railway (see §3); implement as explicit UNAVAILABLE slots.
- §27 Market speed, §31 setup lifecycle, §32 signal stability,
  §40/§41 opportunity clustering, §42-§49 lead-time/actionability/execution
  model, §56 final real-time revalidation, §58/§59 confidence tiers,
  §62 live signal supervision, §64 actionable-signal-rate KPI,
  §66 counterfactual timing, §70 feature ablation, §77 clock sync,
  §81 A+/A/NO TRADE labels.
- §5 WebSocket stream — no reliable stream for XAUUSD on Railway
  (background worker, proxy fallback already in place). Polling loop is
  kept; latency metrics are added instead.

---

## 3. Data-availability reality (honesty constraint)

Available today on Railway: XAUUSD OHLCV (yfinance → PAXG proxy chain),
DXY (yfinance), Finnhub calendar (opt-in key), DeepSeek, Telegram.
NOT available (no free reliable source, no keys): real yields/TIPS,
rate futures, COT, futures OI, options chains, ETF flows, order book,
true tick volume (proxy volume is PAXG token flow — already flagged via
`trust_volume`).

Spec rules we honour: never fabricate; unavailable features are marked
UNAVAILABLE and earn no contribution points. Phases that depend on
blocked data are deferred and listed as slots, not skipped silently.

---

## 4. Implementation phases

Each phase: build → unit/integration tests → full suite green → commit →
push gate. No V2 behaviour changes without a test proving parity.

### Phase A — Timestamp integrity & latency (§4 gap, §5 metrics, §77) — ✅ DELIVERED
- `snapshot.py`: per-cycle `data_latency_ms` (fetch duration) +
  `data_age_s` (age of freshest candle, replay-anchored); both ride the
  LLM dict into every stored signal record (§85).
- `quality.py`: `validate_candles` records `age_s` and flags market
  timestamps ahead of the local clock beyond `data_clock_tolerance_s`
  as `clock_skew` (degraded).
- `notify/telegram.py`: `last_latency_ms` measured per send.
- `main.py`: per-cycle latency line (data / processing / telegram / age).
- Tests: age recording, replay anchoring, clock-skew flag, tolerance
  pass, snapshot metrics, telegram latency (success + failure).

### Phase B — Liquidity map + VWAP (§9, §12)
- `data/liquidity.py` (new): previous day/week/month highs/lows, Asia/
  London/NY session ranges, equal highs/lows, swing clusters; distance
  to nearest liquidity above/below; LIQUIDITY_QUALITY 0-1.
- `data/vwap.py` (new): daily + session VWAP from OHLCV (typical price),
  distance %, reclaim/rejection flag, VWAP trend.
- Wire into `gold_context`/`snapshot`; add `liquidity` + `vwap` context
  to signal records and both Telegram formats (VWAP line).
- Tests: level extraction, distance maths, VWAP calc, reclaim detection.

### Phase C — Location quality + room-to-target (§28, §29)
- `fusion/location.py` (new): LOCATION_QUALITY 0-1 from HTF structure,
  liquidity distances, VWAP, premium/discount, FVG/OB proximity.
- `fusion/room.py` (new): room-to-target vs opposing liquidity; reject
  when room < spread + slippage + min RR (new NoTradeReason
  INSUFFICIENT_ROOM in `risk/engine.py`).
- Add both to setup_quality components + contribution + Telegram.
- Tests: location scoring cases, room rejection cases.

### Phase D — Market speed + displacement/trigger quality (§10, §27, §30)
- `data/speed.py` (new): SLOW/NORMAL/FAST/EXTREME from range-per-minute
  vs ATR, candle formation speed, volatility acceleration.
- `structure.py`: DISPLACEMENT_QUALITY 0-1 (range/ATR, body ratio,
  consecutive candles, BOS/FVG follow-through).
- `fusion/trigger.py` (new): TRIGGER_QUALITY vs TRIGGER_SPEED; trigger
  confirmed/not state.
- Speed feeds actionability limits (max chase) and rejection when EXTREME
  (new NoTradeReason ABNORMAL_SPEED).
- Tests: speed classes, displacement scoring, trigger separation.

### Phase E — Opportunity clustering + signal dedup/suppression (§31, §40, §41, §52, §53, §79)
- `store/opportunity.py` (new): OPPORTUNITY_ID grouping by direction +
  structure event + time proximity; lifecycle states FORMING→…→EXPIRED;
  proposal dedup (same opportunity, same direction, close price, short
  window → suppress duplicate), weaker-opportunity suppression.
- Signal decision states in schema; NoTradeReason OPPORTUNITY_ACTIVE /
  SIGNAL_DUPLICATE.
- Tests: clustering, dedup window, lifecycle transitions.

### Phase F — Signal stability + final real-time revalidation (§32, §56)
- `fusion/stability.py` (new): STABLE/FRAGILE/VERY_FRAGILE via score,
  entry, stop sensitivity to ±1 tick/±1 candle perturbations.
- Orchestrator: final refresh (price/spread/age) after all gates and
  before send; revalidate entry/stop/target; abort send if invalid
  (state change recorded, no Telegram).
- Tests: sensitivity classification, revalidation abort path.

### Phase G — Timing, actionability, execution model (§42-§49, §64)
- `fusion/timing.py` (new): TIMING_QUALITY 0-1 (trigger maturity,
  speed, remaining room, drift, lifecycle); SIGNAL_LEAD_TIME;
  ACTIONABILITY_DEADLINE; EXPECTED_EXECUTION_PRICE/DRIFT from
  `user_reaction_seconds` + `telegram latency` + spread + speed.
- New NoTradeReason TOO_LATE when deadline exceeded pre-send.
- KPI ACTIONABLE_SIGNAL_RATE in dashboard + audit.
- Telegram: execution zone, max chase, valid-until, deadline, timing
  quality, expected drift.
- Tests: timing scoring, deadline math, KPI aggregation.

### Phase H — Confidence tiers + A+/A/NO TRADE (§58, §59, §81)
- Tier assignment from calibrated confidence + sample size (HIGH needs
  calibration_min_samples; LOW → reject). Tiered size caps inside
  risk/engine.py (absolute limits unchanged).
- A+/A/NO TRADE labels in records + Telegram (A+ = top structural +
  liquidity + timing + statistical bucket).
- Tests: tier rules, sizing caps, label mapping.

### Phase I — Live signal supervision + human execution feedback (§62, §63)
- After send: supervise proposal each tick until approved/expired —
  ENTRY STILL VALID / DO NOT CHASE / INVALIDATED / EXPIRED states;
  Telegram follow-ups only on state CHANGE (dedup'd, no spam).
- `approve` command: capture execution price + timestamp; compute user
  latency vs signal; store; EMA of user latency feeds Phase G model.
- Tests: supervision transitions, latency capture.

### Phase J — Forensics + counterfactual timing + rejected-setup outcomes (§65, §66, §67)
- Post-signal snapshots at 1m/3m/5m/10m/30m (from loop, stored).
- Counterfactual timing analysis (research-only, `analytics/`): what-if
  T±3/5/10s entries from M1 data where available.
- Rejected setups: track expected direction vs realized N-candle outcome;
  rejection-quality stats (did we reject correctly?) in dashboard.
- Tests: snapshot persistence, counterfactual pure-function cases.

### Phase K — System health + clock + auto-protection (§76, §77, §78)
- `ops/health.py` (new): DB health, Telegram delivery success rate,
  AI latency, provider status, queue/tick lag, clock offset; health
  score; BLOCK NEW SIGNALS on health failure (kill-switch route).
- Tests: health score thresholds, auto-block.

### Phase L — Realistic backtesting + feature ablation (§69, §70, §72)
- `backtest/engine.py`: human reaction delay + telegram latency + drift
  + slippage model (shared with Phase G).
- `analytics/ablation.py` (new): WITH vs WITHOUT feature-group harness
  over the same walk-forward window (SMC, DXY, VWAP, liquidity, speed…).
- Tests: delay model, ablation runner.

### Deferred slots (data-blocked, honesty-marked)
- `data/macro.py`, `data/positioning.py`, `data/options.py`,
  `data/flows.py`, `data/yields.py`: schema + UNAVAILABLE status only,
  zero contribution, surfaced in Telegram "données indisponibles". Activate
  behind explicit API keys later. Volume profile (§13) waits for a
  trustworthy volume source; VWAP (Phase B) ships first.

---

## 5. Constraints carried from V2 (non-negotiable)

- Human-in-the-loop: robot never executes alone; approve/reject stays.
- Deterministic risk engine is final authority; AI is analyst only.
- One coherent snapshot per cycle; fail-closed on data quality.
- Every decision (incl. rejections) stored with gate trail + version.
- Rejections keep reaching Telegram with full analysis (user requirement).
- No look-ahead, no fabricated data, no fake neutral contributions.
- Railway background worker, proxy fallback, single production instance.

## 6. Risks

- Latency: each new engine adds per-cycle compute — keep all new
  calculations vectorized over the cached snapshot; no per-tick work
  beyond Phase I supervision (cheap state checks).
- Data honesty: phases using proxy data must respect `trust_volume`.
- Regression: 394 tests + A/B `ab-compare` gate on any scoring change
  (V2 verdict must stay IMPROVED, or the change is reverted).
