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

### Phase B — Liquidity map + VWAP (§9, §12) — ✅ DELIVERED
- `data/liquidity.py` (new): previous day/week/month highs/lows, Asia/
  London/NY session ranges, equal highs/lows, swing clusters; distance
  to nearest liquidity above/below; LIQUIDITY_QUALITY 0-1
  (0.5×proximity + 0.3×density + 0.2×freshness); levels deduped,
  PMH/PML via the 1d frame.
- `data/vwap.py` (new): daily + session VWAP from OHLCV (typical price),
  distance %, reclaim/rejection flag, VWAP trend; marked UNAVAILABLE
  with the reason when volume is untrusted (proxy) or absent.
- Wired into `snapshot.py`: `MarketSnapshot.liquidity`/`vwap` fields,
  included in `entry_snapshot_for_llm` (→ every signal record) and
  `context_for_risk`; `trust_volume` hoisted and shared with the shock
  engine.
- `notify/telegram.py`: VWAP state + nearest liquidity lines added to
  both proposal and rejection formats (`_liquidity_vwap_lines`).
- Tests: level extraction, distance maths, quality scoring, PMH/PML,
  session windows, VWAP calc, reclaim/rejection/trend, trust gating,
  snapshot wiring, proxy unavailability.

### Phase C — Location quality + room-to-target (§28, §29) — ✅ DELIVERED
- `fusion/location.py` (new): LOCATION_QUALITY 0-1 as the mean of four
  side-aware sub-scores — liquidity proximity (ATR bands), VWAP
  relation (reclaimed/above/below/rejected mirrored per side),
  premium/discount vs the session-range midpoint, and untested
  FVG/OB shelter on the stop side. Duck-typed on the Phase B snapshot
  blocks; neutral 0.5 when data is absent (never fabricated).
- `fusion/room.py` (new): room-to-target vs the opposing liquidity pool
  (nearest_above for LONG, nearest_below for SHORT); spread + slippage
  subtracted from the raw distance before the R conversion; reject when
  the after-cost room < `room_min_rr` R. New `NoTradeReason`
  INSUFFICIENT_ROOM wired into `risk/engine.py` (`_room_gate`, after the
  no-trade gates; `room_gate_enabled`/`room_min_rr` settings). No mapped
  opposing level never blocks (data honesty, spec §4).
- Both ride the existing plumbing: location is the 8th setup-quality
  component (WEIGHTS rebalanced to sum 1.0) and joins the §47
  contribution labels; the room rejection renders with its reason + code
  in the Telegram rejection format.
- Tests: location sub-score bands/mirroring/zone preference (6),
  compute_room math + engine gate (12, incl. gate on/off, min RR,
  spread from fusion, no-data pass), setup-quality weight formula;
  backtest fixture gains deterministic pullbacks so the gate sees
  realistic distances (chase-only monotonic series correctly trades 0,
  documented by a dedicated test).

### Phase D — Market speed + displacement/trigger quality (§10, §27, §30) — ✅ DELIVERED
- `data/speed.py` (new): SLOW/NORMAL/FAST/EXTREME from range-per-minute
  vs ATR, candle formation speed (typical-range fraction consumed,
  normalized by elapsed candle time with a 0.5 floor so a fresh
  candle's first ticks never score more than 2x), and volatility
  acceleration (ATR now vs `accel_lookback` ago). EXTREME = ratio >=
  `extreme_mult` or FAST ratio + acceleration lift; SLOW = formation
  <= `slow_mult` with ATR not expanding; insufficient history fails
  open to NORMAL (unknown, not fast).
- `structure.py`: DISPLACEMENT_QUALITY 0-1 on the latest displacement
  candle (0.35 range/ATR, 0.25 body ratio, 0.2 consecutive candles,
  0.2 BOS/FVG follow-through); None when no displacement exists
  (honest, never fabricated).
- `fusion/trigger.py` (new): TRIGGER_QUALITY (0-1 mean of BOS support,
  sweep/reclaim, displacement, zone shelter) with TRIGGER_SPEED kept
  separate; `confirmed = quality >= trigger_confirm_min and speed !=
  EXTREME`. Side-aware, computed in `build_fusion_context` and riding
  `FusionContext.trigger` into signal records + Telegram.
- `risk/engine.py`: opt-in `no_trade_extreme_speed` gate — EXTREME
  speed rejects new entries with the new NoTradeReason ABNORMAL_SPEED
  (mirrors HIGH_VOLATILITY); the gate trail always records the state.
- `notify/telegram.py`: Vitesse (state + ratio) and Déclencheur
  (confirmé / non confirmé) lines in both formats.
- Tests: speed classes/formation/acceleration/honesty (9), trigger
  separation/confirmation (8), displacement scoring (3), engine gate
  (3), snapshot wiring (1).

### Phase E — Opportunity clustering + signal dedup/suppression (§31, §40, §41, §52, §53, §79) — ✅ DELIVERED
- `store/opportunity.py` (new): deterministic OPPORTUNITY_ID =
  `{SYMBOL}:{TF}:{side}:{anchor}`; anchor priority = most recent
  same-direction BOS → same-direction displacement → trigger-side
  liquidity pool (lows for LONG, highs for SHORT); NEUTRAL side or
  unreadable structure → no anchor → dedup never blocks (honesty).
- Dedup verdict: a pending stronger proposal for the same opportunity
  suppresses a weaker re-signal (OPPORTUNITY_ACTIVE) while a stronger
  one supersedes at save time; any recent proposal (dedup window,
  default 180m) whose stored snapshot price is within the tolerance
  band (default 0.1%) of the current price → SIGNAL_DUPLICATE; window
  and price comparisons run in Python with tz-normalized timestamps;
  disabled config, no anchor or a DB failure fails open (never blocks).
- Lifecycle: `Opportunity` table row upserted every cycle — FORMING →
  TRIGGERED (with trigger_signal_id, never downgraded) → EXPIRED when
  un-triggered past `opportunity_ttl_minutes` (default 720).
- `schema/types.py`: `DecisionState` enum (PENDING/APPROVED/REJECTED/
  EXPIRED/SUPERSEDED); `SignalProposal.status` now uses it; actions
  (save/decide/revert) read/write the enum values.
- `fusion/types.py`: NoTradeReason OPPORTUNITY_ACTIVE, SIGNAL_DUPLICATE.
- `orchestrator.py`: dedup gate between fusion (side known) and the
  risk engine — rejection carries the no_trade_reason and a gate-trail
  entry; every signal record is stamped with `opportunity_id` and
  `opportunity_state` (TRIGGERED for proposals, FORMING for rejections;
  stamping is forensics and runs even with dedup disabled);
  `_run_pipeline` returns the opportunity context as an 8th element.
- `store/db.py`: additive SQLite migration for `signals.opportunity_id`
  + `signals.opportunity_state`.
- Tests: anchor/id derivation (5), dedup verdict window/price/pending
  semantics (10), lifecycle transitions (2), orchestrator gate + record
  stamping + disabled-config honesty (3).

### Phase F — Signal stability + final real-time revalidation (§32, §56) — ✅ DELIVERED
- `fusion/stability.py` (new): STABLE/FRAGILE/VERY_FRAGILE from two
  deterministic perturbations — ±1 tick on price recomputes the setup-
  quality score (score sensitivity = max delta), −1 candle recomputes
  ATR + speed on the shortened frame (entry sensitivity = second-to-
  last close vs price in ATRs; stop sensitivity = stop-distance move in
  ATRs; a speed-state flip marks a regime boundary). Worst axis wins;
  NEUTRAL is not classified; insufficient history is FRAGILE (unknown
  is not stable) — enrichment stored with every signal record, never a
  hard gate by itself.
- `FusionContext.stability` computed in `build_fusion_context`; the
  orchestrator now persists `trigger` + `stability` in the signal
  record's fusion dict (fixes the Phase D trigger line never
  rendering — Telegram reads it from the record).
- Final real-time revalidation (§56): after all gates and before the
  send, one fresh tick must still support the proposal — price drift
  vs entry (`revalidate_max_drift_pct`), fresh spread vs the existing
  BAD_SPREAD ceiling and tick age (`revalidate_max_age_s`). A failure
  returns a recorded Rejection (new NoTradeReason SIGNAL_INVALIDATED;
  spread failures reuse BAD_SPREAD) with a `revalidation` gate-trail
  entry and a `signal_invalidated_pre_send` audit event — the send
  never happens (main only sends on SignalProposal).
- Tick sources: `MT5Broker.tick` (broker ask/bid/tick-time), ccxt
  `MarketData.tick`, `GoldData.tick` (MT5 delegate live, PAXG ticker on
  proxy). No tick source, no data or a fetch error fails open — no
  data never blocks (spec §4).
- Tests: stability classification (neutral/insufficient/stable/fragile/
  very-fragile/speed-flip/engine wiring, 8), revalidation gate
  (pass/drift/spread/stale/disabled/no-source/fetch-failure, 7), tick
  sources (5).

### Phase G — Timing, actionability, execution model (§42-§49, §64) ✅ DELIVERED
- `fusion/timing.py` (new): TIMING_QUALITY 0-1 from five deterministic
  components — trigger maturity (structure-event freshness), speed
  (SLOW 1.0 / NORMAL 0.8 / FAST 0.5 / EXTREME 0.2), remaining room
  (room-to-target in R, normalized), drift (expected move during the
  reaction window + spread cost vs stop distance), lifecycle
  (opportunity age vs TTL). Missing axes score a neutral 0.5, never
  fabricated.
- SIGNAL_LEAD_TIME from the speed-adjusted pace (range/ATR per minute ×
  state multiplier), capped at the opportunity TTL;
  ACTIONABILITY_DEADLINE = now + lead (epoch + ISO).
- EXPECTED_EXECUTION_PRICE/DRIFT from `user_reaction_seconds` +
  `telegram_latency_s` + spread + speed; execution zone (entry ±
  `max_chase_atr_mult` × ATR) + max chase exposed to Telegram.
- New NoTradeReason TOO_LATE: orchestrator pre-send gate after Phase F
  revalidation aborts the send when lead ≤ reaction window (recorded
  rejection + `signal_too_late_pre_send` audit; uncomputable pace
  fails open). Timing payload stamped on the signal record.
- KPI ACTIONABLE_SIGNAL_RATE in the dashboard (proposals / (proposals
  + TOO_LATE), today) + audit trail; new "Signaux actionnables" card.
- Telegram: timing quality + label, execution zone, max chase, expected
  price (drift), valid-until deadline.
- Tests: timing scoring (10), lead/deadline math (6), opportunity age
  helper (2), orchestrator TOO_LATE gate (4), Telegram lines (2),
  dashboard KPI aggregation (3).

### Phase H — Confidence tiers + A+/A/NO TRADE (§58, §59, §81) ✅ DELIVERED
- `fusion/tier.py` (new): ConfidenceTier HIGH/MEDIUM/LOW from the
  calibrated win rate (`tier_high_min_calibrated` 0.6,
  `tier_low_max_calibrated` 0.45); uncalibrated data is MEDIUM —
  honest, never a block, never a promotion (spec §4/§21). Size
  fractions: HIGH 1.0, MEDIUM `tier_medium_size_cap` (0.75), LOW 0.0.
- Risk engine tier gate: LOW (a sufficient calibrated sample whose
  bucket loses money) is refused with the classified NoTradeReason
  LOW_TIER, right after the statistical-quality gate; the tier mark
  always rides the gate trail. The tiered size cap is applied in
  sizing; absolute exposure / positions / notional limits unchanged.
- A+/A/NO TRADE labels (deterministic): A+ only when tier is HIGH AND
  structural setup ≥ `a_plus_setup_quality_min` (0.7) AND timing
  quality ≥ `a_plus_timing_min` (0.7) AND liquidity room ≥
  `a_plus_room_min_r` (2.0 R); any other approved proposal is A;
  rejections are NO TRADE. Stamped on the signal record (`signal_label`
  column + `fusion.tier`, additive SQLite migration) and rendered in
  Telegram on proposals (with palier) and rejections.
- Tests: tier rules (6), risk-engine gate + sizing caps (5),
  orchestrator stamping (3), Telegram label lines (4).

### Phase I — Live signal supervision + human execution feedback (§62, §63) ✅ DELIVERED
- `supervision.py` (new): pure per-tick classification of pending
  proposals — VALID / DO_NOT_CHASE (price beyond the execution zone) /
  INVALIDATED (stop or target reached) / EXPIRED (actionability
  deadline passed), priority EXPIRED > INVALIDATED > DO_NOT_CHASE.
  No deadline never expires (spec §4).
- The loop supervises every tick: state changes persist once (unchanged
  ticks are a no-op — no DB write, no spam) and return exactly one
  Telegram follow-up per change; the first None→VALID classification
  is silent. INVALIDATED/EXPIRED move the proposal out of `pending` —
  approving a dead signal is refused; DO_NOT_CHASE stays pending
  (pullback entries remain possible).
- The orchestrator stamps the timing payload's deadline + max-chase
  zone on every proposal (`actionability_deadline` / `max_chase`, new
  proposal columns, additive migration) so supervision has its inputs.
- §63: `approve` measures the signal-to-execution latency and stores it
  (`user_latency_s`); its EMA (`user_latency_ema`, span
  `user_latency_ema_span`) overrides the Phase G reaction window in
  `compute_timing` — the timing model learns the trader's real
  reaction time (fail-open: no samples → configured budget).
- Tests: state machine (12), persistence + dedup (4), latency EMA +
  timing override (2), orchestrator stamping + EMA feed (2), Telegram
  follow-up lines (2).

### Phase J — Forensics + counterfactual timing + rejected-setup outcomes (§65, §66, §67) ✅ DELIVERED
- `store/models.py`: `PostSnapshot` table (signal_id, offset_min, ts,
  price, high, low; unique (signal_id, offset_min)) — a new table, so
  `create_all` creates it on existing databases (no additive column
  migration needed).
- `forensics.py` (new): `SnapshotRegistry` + `capture_due` — every
  decision (proposal OR rejection) registers its signal_id + ts and
  the loop samples the current candle at 1/3/5/10/30 min, persisting
  one row per offset. `due()` is purely observational and `mark()`
  confirms persistence, so a tick for another symbol never eats an
  owed capture; fully-captured or >32-min-old entries are pruned.
  Restarts re-register the last 31 minutes of decisions (`restore()`),
  and an already-persisted row is marked, never re-inserted.
- `analytics/counterfactual.py` (new, research-only): what-if entries
  at T±10/±5/±3/0 s from the M1 OHLCV — linear open→close
  interpolation inside the bar containing the shifted timestamp
  (`linear_interpolation_m1`, a labeled approximation, never
  tick-accurate); data gaps return available=False. `counterfactual_rr`
  shifts only the entry (stop/target stay plan-based) and returns None
  when the shifted entry crosses the stop.
- `analytics/rejected.py` (new): every rejection with a directional
  score and a 30m snapshot is classified CORRECT_REJECT (the market
  moved against the refused direction) / WRONG_REJECT (moved at least
  `rejection_move_threshold_pct`, a percent) / INCONCLUSIVE (the noise
  band); `rejection_quality` reports correct/(correct+wrong) — the
  inconclusive band never inflates it (spec §36). Dashboard section
  “Qualité des rejets — 30 min après (§67)” with correct/wrong/
  inconclusive cards + the honest ratio.
- Config: `forensics_enabled` (loop gate) and
  `rejection_move_threshold_pct` (0.05).
- Tests: registry due/mark/prune/restore + symbol-filtered capture +
  idempotent persistence (16), counterfactual interpolation/gaps/RR
  (10), rejection classification/aggregation/dashboard pipeline (13).

### Phase K — System health + clock + auto-protection (§76, §77, §78) ✅ DELIVERED
- `ops/health.py` (new): six vitals — DB reachability (a real
  round-trip), Telegram delivery rate + last latency, AI availability
  (real answers vs heuristic fallbacks), provider data-quality,
  queue/tick lag (processing vs tick interval), and clock offset
  (Phase A skew flag) — fused into one 0-100 score with renormalized
  weights: checks that do not apply (Telegram disabled, no engine, no
  cycle yet) are EXCLUDED, and an empty score is 100 (blocking on a
  vacuum would be dishonest, spec §4).
- Status bands: >= `health_ok_score` HEALTHY, < `health_block_score`
  CRITICAL, between = DEGRADED. A CRITICAL streak of
  `health_block_consecutive` evaluations engages the kill-switch route
  (`RiskEngine.halt` → new signals blocked, audit + Telegram alert); a
  HEALTHY streak of `health_recover_consecutive` auto-resets ONLY the
  halts this monitor engaged — a manual halt is never touched.
- The loop observes every tick (evaluated at `health_eval_interval_s`,
  startup baseline immediately) and prints the score with a colour
  tone; Telegram alerts fire on engage/recover only.
- `TelegramNotifier` gains delivery counters (`sends`/`failures`) for
  the health score.
- Tests: six checks (12), weighted score + renormalization + empty
  (3), status boundaries (2), monitor engage/recover/manual-halt-
  sacred/streak-reset/interval-gating (5).

### Phase L — Realistic backtesting + feature ablation (§69, §70, §72) ✅ DELIVERED
- `backtest/engine.py`: opt-in realistic execution
  (`backtest_realistic_execution`): the fill is scheduled at signal
  time + the SAME human reaction window Phase G assumed (the signal's
  stored timing `reaction_s`, fallback `user_reaction_seconds` +
  `telegram_latency_s`) and priced with the Phase G drift projection
  (pace-based drift from the stored payload) plus the configured
  `slippage` on top of the broker spread; the position is managed from
  the first candle that opens after the fill. Reaction windows longer
  than the timeframe defer the fill across candles — no look-ahead
  ever: the projection comes from the signal's own payload, never
  from future candles. Off = the deterministic §24/§25 next-open
  baseline (the A/B baseline contract is untouched).
- Feature ablation toggles in the canonical snapshot:
  `feature_smc/dxy/vwap/liquidity/speed_enabled` — off removes the
  group's inputs (empty structure map, no DXY gauge/context, no VWAP,
  no liquidity map, no speed state); every consumer answers its
  documented neutral, no data never blocks (§4).
- `analytics/ablation.py` (new): WITH vs WITHOUT harness — the same
  window replayed per group on isolated databases, verdicts by the
  §41 rules (IMPROVED = the group cost more than it gave on this
  window; WORSE = it earned its place; MIXED/INSUFFICIENT_DATA = no
  conclusion).
- `analytics/sample_size.py` (new): the §36 sample-size derivation —
  `required_trades` (effect size, variance, power -> N per side) and
  `detectable_effect` (the inverse floor); claims below the floor are
  labeled UNVALIDATED, never presented as evidence.
- Held-out final validation (§7): DATA-BLOCKED — needs the locked
  12-month XAUUSD window fetched once, at the very end of the phase;
  marked here, not skipped silently.
- Tests: snapshot toggles on/off (2), reaction-fill scheduling/drift/
  fallback (4), realistic replay delay/deferral/slippage/determinism
  (4), ablation harness structure/end-to-end/verdicts/determinism/
  unknown-group (5), sample size (5).

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
  beyond Phase I supervision and Phase J snapshot captures (cheap
  registry checks).
- Data honesty: phases using proxy data must respect `trust_volume`.
- Regression: 664 tests + A/B `ab-compare` gate on any scoring change
  (see §7 for what IMPROVED means and when it can fire).

## 7. Evaluation honesty protocol (added after external review)

Tests prove code correctness, not trading edge. Edge is a separate
claim, gated on sample size and a truly untouched validation set:

- **Held-out final validation set**: a fixed date window of XAUUSD
  history that is NEVER used for development, tuning, walk-forward or
  A/B. It is touched exactly once, at the end of Phase L, to answer a
  single question: does the calibration curve hold out-of-sample?
  Default: the 12 months ending at the Phase C model freeze; all
  development/walk-forward/A-B windows must end before it. Locked
  now, so no later phase can peek.
- **What IMPROVED means**: the A/B verdict (`analytics/compare.py`,
  §40/§41) requires ≥ `min_trades` resolved trades per side, better
  expectancy AND drawdown/profit factor not meaningfully worse;
  fewer trades alone is never an improvement (§41). Today the paper
  DB has 0 resolved trades, so the gate reports INSUFFICIENT_DATA by
  design — never IMPROVED. Until trades exist, the operative
  regression check per phase is: full suite green + walk-forward
  parity + no WORSE verdict.
- **Sample size (§36)**: no fixed "100 trades" rule — the required N
  for any performance claim is derived from effect size, variance and
  power (`analytics/sample_size.py`, shipped in Phase L):
  `required_trades` and the inverse `detectable_effect` floor. Until
  the sample supports the effect, every confidence/win-rate number
  shown to the trader is labeled UNVALIDATED, not presented as
  probability.
- **Current frontier**: all implementation phases (A–L) are
  DELIVERED — the V-MONSTER spec's 92 sections are covered, the
  deterministic risk engine is the final authority, and the bot runs
  human-in-the-loop end to end. The last phase (L) added realistic
  backtests that replay the trader's real reaction latency
  (drift-projected fills, not idealised next-open) and the ablation
  harness measuring what each feature group (SMC / DXY / VWAP /
  liquidity / speed) actually contributes on a window, with
  §41-honest verdicts. The one item left open is the held-out final
  validation (§7): still DATA-BLOCKED, waiting on the locked 12-month
  XAUUSD window fetched once. Confidence scores have no calibration
  guarantee until that run and until samples pass the §36 power
  floors.
- **Proxy VWAP honesty**: on proxy volume (PAXG token flow) the VWAP
  is marked unavailable in Telegram with the reason — it is never
  presented as institutional gold VWAP.
