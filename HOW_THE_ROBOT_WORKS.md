# How Your Trading Robot Thinks — User-Friendly Guide (INTELLIGENCE_V2)

A step-by-step explanation of the XAUUSD robot's decision process, written
to help you judge its signals and improve its accuracy. Updated to reflect
the full INTELLIGENCE_V2 upgrade (8 phases, 57 spec sections).

---

## 1. The Big Picture

```
 MARKET DATA            3 DEEPSEEK BRAINS        THE JUDGE            YOU
 (every 15 min)         (research & vote)     (12 hard gates)      (decide)
┌──────────────┐      ┌───────────────────┐  ┌──────────────────┐  ┌───────┐
│ 15m candles  │─────▶│ 1. Technical      │─▶│ Kill-switch      │─▶│Telegram
│ 4h candles   │      │ 2. Sentiment      │  │ Neutral/confid.  │  │signal │
│ DXY gauge    │      │ 3. Regime         │  │ News blackout    │  │  ↓    │
│ SMC structure│      └───────────────────┘  │ Shock + cooldown │  │ MT5   │
│ Shock meter  │         │ weighted vote     │ No-trade zones   │  │ (you) │
│ News calendar│         ▼                  │ Statistical qlty │  └───────┘
└──────────────┘   side + confidence ──────▶│ DXY / HTF bias   │
      ▲                                     │ Positions/Sizing │
      │                                     └──────────────────┘
      │      ┌───────────────────────────┐            │
      └──────│ Self-measurement loop     │◀───────────┘
             │ (stats, A/B, replay)      │
             └───────────────────────────┘
```

The key design rule: **the AI proposes, the machine disposes.**
DeepSeek never decides anything final — the deterministic risk engine
holds the final authority and cannot be overruled. Every decision (pass
or fail) is stored with the full trail of gates, so you can always see
*exactly* why a signal lived or died.

---

## 2. Step 1 — The Eyes: Gathering Data (every new 15m candle)

| Data | Source | Notes |
|---|---|---|
| XAUUSD 15m candles | yfinance gold futures → PAXG/USD fallback (24/7) | Last 300 candles |
| XAUUSD 4h candles | same sources | For the trend bias |
| DXY gauge (0–100) | MT5 `DXY_U6` → yfinance `DX-Y.NYB` fallback | 100 = weak dollar = bullish gold |
| Indicator snapshot | computed locally (pandas) | EMA 20/50/200, RSI, MACD, ATR, ADX, Bollinger, volume |
| **SMC market structure** | computed locally | Break of Structure, CHoCH, Fair Value Gaps, order blocks |
| **Shock meter** | computed locally | Detects violent candles: NORMAL / volatility expansion / SHOCK |
| **Economic calendar** | real provider only (Finnhub, opt-in) | Upcoming USD events with importance + minutes to go |

All numbers are computed **deterministically by our code** — the AI
never calculates math, it only reads the results. The calendar is the
single exception that comes from outside, and it comes from a **real
data provider, never from the AI** — DeepSeek is forbidden from
inventing news.

---

## 3. Step 2 — The Brains: How DeepSeek "Researches"

**Honest definition first:** DeepSeek does NOT browse the internet, read
news, or see charts. It receives a **structured JSON snapshot of numbers**
and reasons over it using its trained pattern knowledge. Each of the 3
agents gets a strict mission:

### Agent 1 — Technical (weight 45%)
- **Sees:** all indicator values + the 4h bias context
- **Mission:** "Is there a tradeable setup right now?"
- **Returns:** `bias` (long/short/neutral), `conviction` (0–1), support,
  resistance, 2–3 sentences of reasoning
- **Rule:** must justify everything from the numbers given; never invents
  prices, events or news

### Agent 2 — Sentiment (weight 20%)
- **Sees:** the DXY gauge + price context
- **Mission:** "What is the dollar doing to gold right now?"
- **Returns:** `score` (−1 to +1), `tone` (risk-on/risk-off/neutral)
- **Recall:** gauge 100 = dollar very weak = gold should rise

### Agent 3 — Regime (weight 35%)
- **Sees:** ADX, ATR percentile, EMA alignment
- **Mission:** "What kind of market is this?"
- **Returns:** `regime` (trending up / trending down / ranging /
  high volatility) + `trend_strength`

Every answer must be **valid JSON** matching our schema. If DeepSeek
fails, is unreachable, or returns garbage → the agent falls back to a
simple local heuristic, **clearly labelled** in the logs and alerts you
("degraded mode").

**Cost control (new in V2):** before any DeepSeek call, the robot runs
free local checks — kill-switch armed? price/ATR sane? spread below the
configured ceiling? If the market isn't worth analyzing, the 3 AI calls
are skipped entirely. Your API balance is never spent on garbage data.

---

## 4. Step 3 — The Vote: Fusion + Setup Quality

The three verdicts are merged into one number:

```
score = 0.45 × (technical side ±1) × conviction
      + 0.20 × sentiment score
      + 0.35 × (regime side ±1) × trend_strength

score ≥ +0.25  →  LONG
score ≤ −0.25  →  SHORT
otherwise      →  NEUTRAL (no signal)

confidence = |score| (capped at 1.0)
```

On top of the raw vote, V2 adds three quality layers before the judge:

1. **Setup quality** — a 0–1 score checking the context around the
   signal: is the structure aligned (valid Break of Structure in the
   trade direction)? Is price near a sensible zone (order block /
   FVG)? A low score can refuse the signal even if confidence is high.
2. **Conflict detection** — do the agents actually agree? Technical
   long + regime ranging + dollar ambiguous → `CONFLICTED`; the robot
   refuses instead of trading on disagreement.
3. **Calibration** — the raw confidence is adjusted using real
   historical outcomes ("did signals with 0.70 confidence really win
   70% of the time?"). The calibrated number is what the judge uses.

---

## 5. Step 4 — The Judge: Risk Engine Hard Gates (in order)

If ANY gate fails, the signal is refused with a written reason.
The full list — new V2 gates in **bold**:

| # | Gate | Rule |
|---|---|---|
| 1 | Kill-switch | If halted (daily loss −3% / drawdown −10% / equity 0), everything is refused |
| 2 | Neutral | Fused signal must have a direction |
| 3 | Confidence | `confidence ≥ 0.55` |
| 4 | **News blackout** | (opt-in) HIGH-importance USD event within 30 min → refused (NFP, CPI...) |
| 5 | **Market shock** | Violent candle (range/ATR/volume/spread explosion) → new entries blocked + 30 min cooldown. Volatility expansion = warning only. Volume counts only on real gold data — ignored on the PAXG proxy, where token volume measures crypto flow, not gold |
| 6 | **No-trade zones** | Structure against the trade, consolidation, agent conflict, low setup quality, bad spread, extreme volatility |
| 7 | **Statistical quality** | (opt-in) similar historical signals with negative expectancy → refused |
| 8 | **DXY concurrency** | LONG needs gauge ≥ 55 (weak dollar); SHORT needs gauge ≤ 45 (strong dollar) |
| 9 | **HTF bias** | LONG only in 4h uptrend; SHORT only in 4h downtrend; choppy 4h (ADX < 20) blocks everything |
| 10 | Duplicates | One open position per symbol; max 3 total |
| 11 | Validity | Price and ATR must be sane |
| 12 | Sizing | Risk = 1% of equity; size = risk ÷ stop distance; exposure capped at 25% of equity; dust (< $5 notional) refused |

**SL & TP are mechanical:** stop = entry ∓ 2×ATR, target = entry ± 4×ATR
(always a 2:1 reward/risk).

---

## 6. Step 5 — The Output

A passing signal becomes a **proposal**:
- saved to the database **with the full gate trail** — every gate,
  pass or fail, with its detail (inspect it with `signal <id>`)
- sent to your phone: entry, stop, target, size, confidence, the DXY
  line and the 4h bias line
- printed in the console with an `approve` command

After the trade closes, the robot records the **outcome** (win/loss,
MFE/MAE — how far the trade went in your favor before ending) and feeds
it into the statistics engine. Nothing is executed automatically — **you
remain the decision-maker**.

---

## 7. The V2 Upgrade: The Robot Now Measures Itself

The biggest change in INTELLIGENCE_V2: the robot no longer just
suggests — it **studies its own record**:

| Tool | What it answers | Command |
|---|---|---|
| **Statistics engine** | Win rate, expectancy, calibration per confidence bucket | automatic + dashboard |
| **Replay** | "What exactly did the robot know at that moment?" | `replay <ts>` |
| **Backtest** | "How would this strategy have done on history?" (candle-by-candle, no look-ahead) | `backtest XAUUSD` |
| **Walk-forward** | "Does it hold up out-of-sample, or is it curve-fitted?" | `walkforward XAUUSD` |
| **Sensitivity** | "What if the parameters were slightly different?" | `sensitivity XAUUSD` |
| **Monte Carlo** | "How bad can drawdown get if trades shuffle?" | `montecarlo XAUUSD` |
| **A/B comparison** | "Is V2 really better than the old robot?" | `ab-compare XAUUSD` |
| **Dashboard** | One HTML page: performance, confidence buckets, regime stats, shock & news cards | `dashboard` |

**The first real A/B result** (gold data, Aug 4 → Sep 16 2026, 15m):

| Metric | LEGACY_BASELINE | INTELLIGENCE_V2 |
|---|---|---|
| Trades | 9 | 5 |
| Expectancy | 0.24 R | **0.32 R** |
| Profit factor | 1.26 | **1.38** |
| Max drawdown | 0.36% | **0.17%** |

Verdict: **IMPROVED** — and not just "fewer trades": the tool
explicitly refuses to count trading less as an improvement (§41); the
verdict comes from better expectancy, better profit factor and half the
drawdown.

---

## 8. Simulation A — A Signal That Passes

Monday, 14:30 UTC (London+NY overlap). Gold 2400. ATR 5.

**Agent verdicts:**
- Technical: `long`, conviction 0.60
- Sentiment: gauge 62 (dollar weak) → score +0.24
- Regime: `trending up`, strength 0.70

**Fusion:**
```
0.45 × 0.60  = +0.270
0.20 × 0.24  = +0.048
0.35 × 0.70  = +0.245
score = +0.563  →  LONG, confidence 0.56
```

**Gates:**
1. Not halted ✓
2. Direction LONG ✓
3. Confidence 0.56 ≥ 0.55 ✓
4. News: no blocking event in the window ✓
5. Shock meter: NORMAL ✓
6. Structure: bullish BOS on 15m, price above an order block, setup
   quality 0.7, no agent conflict ✓
7. Statistical quality: not enough similar history yet → "unknown",
   which never blocks (only a proven negative edge blocks) ✓
8. DXY: 62 ≥ 55 ✓ (weak dollar = bullish gold)
9. HTF 4h bias: bull ✓
10. No open position on XAUUSD ✓
11. Price/ATR valid ✓
12. Sizing: risk = 100 USD (1% of 10k) → stop distance 10 → size 10 units;
    exposure cap 25% of equity = 2,500 → shrunk to 2,500 ÷ 2,400 ≈ 1.04 units ✓
    (the position shrinks, so the real money at risk drops to ~1.04 × 10 ≈ 10.40 USD)

**Proposal:** BUY 1.04 @ 2400, SL 2390, TP 2420, confidence 0.56
(planned risk 100 USD — but the 25% exposure cap actually shrinks the position)
→ Telegram message arrives with all details. **You decide.**

---

## 9. Simulation B — Rejected by the DXY Gate

Same agents say LONG with confidence 0.70 — but the gauge is 48.
The dollar is not weak enough. The judge refuses:

> "DXY concurrency: gauge 48 not weak-dollar (need >= 55) for a LONG"

The robot stays flat. This is the gate doing its job.

---

## 10. Simulation C — Rejected by the HTF Bias Gate

Agents say SHORT, confidence 0.65, DXY gauge 30 (strong dollar ✓) —
but the 4h trend is bullish (EMA 50 above EMA 200).
Fighting the 4h trend is a losing trade statistically, so:

> "HTF bias: bull on 4h blocks SHORT"

Refused. "The trend is your friend" — enforced by machine, not by hope.

---

## 11. Simulation D — Rejected by the News Blackout (V2)

Agents say LONG, confidence 0.70, everything aligned — but US CPI is
released in 25 minutes (HIGH importance, USD). If the news filter is
enabled, the judge refuses:

> "news blackout: CPI in 25m (HIGH)"

No matter how good the setup looks, the robot will not step in front of
a scheduled market-moving event. (The gate is opt-in: it only works
when a real calendar provider is configured — see §14.)

---

## 12. Simulation E — Rejected by the Shock Meter (V2)

A violent candle lands: range 3× the normal average, ATR exploding,
volume spiking. The shock detector classifies it **SHOCK**:

> "market shock: range 3.2x baseline (SHOCK)"

New entries are blocked, and a **30-minute cooldown** starts. Even after
the candles calm down, the robot stays flat until the cooldown expires —
shock aftermath (whipsaw, widened spreads) is where accounts die. A mild
volatility expansion, by contrast, only adds a warning line to the
signal record; it doesn't block.

---

## 13. The Loop Over Time (a typical day, 24/7 test mode)

| Moment | What happens |
|---|---|
| Every 60s | Tick: fetch latest data, manage open positions (stop/target) 24/7 |
| Every new 15m candle | Free pre-checks (cost control) → full analysis (3 DeepSeek calls + 12 gates) |
| Candle with no new info | No analysis — no wasted cost |
| SHOCK detected | New entries blocked + cooldown started |
| High-impact news (if enabled) | Blackout window around the release |
| Kill-switch trip | Alert on Telegram, no new proposals |
| Position closes | Outcome + MFE/MAE stored; confidence calibration updates |
| Every day (UTC) | DeepSeek balance check → alert if < 1 USD |
| Every rejected signal | Telegram alert: gate + reason + full agents' analysis (biases, reasoning, contributions, gate trail) |

---

## 14. Honest Limits (updated for V2)

- ✅ **News filter — DONE** (opt-in): real provider only, never
  AI-invented news. It is **OFF by default**: enabling it requires a
  (free) Finnhub API key. When the provider is unreachable, the robot
  fails open — it trades without the filter rather than blocking
  forever, so treat it as a helper, not a guarantee.
- ✅ **SMC structure detection — DONE**: BOS/CHoCH/FVG/order blocks are
  computed locally and used by setup quality + the no-trade gate.
- ❌ No multi-symbol confirmation (e.g., GBPUSD) yet
- ⚠️ Weekends: DXY gauge frozen at Friday's close
- ⚠️ Degraded mode: if DeepSeek is unreachable, heuristics take over
  (you are alerted — treat those signals with extra caution)
- ⚠️ Statistical quality needs history: with an unknown sample the gate
  stays silent (by design, spec §31). Early weeks are less filtered.
- ⚠️ The shock cooldown can keep you flat for 30 min after a violent
  candle — that's deliberate, not a bug.

---

## 15. How to Improve Decision Accuracy (your levers)

1. **Read the rejections.** Every refusal now reaches you on Telegram
   with the gate that fired, the agents' biases and reasoning, and what
   supported/contradicted the setup (consecutive repeats are deduplicated
   into a heartbeat). Use `signal <id>` to see the full gate trail of any
   decision, or `history` for the recent audit events.
2. **Trust the statistics, not feelings.** The dashboard shows win rate
   and expectancy **per confidence bucket**. If 0.70+ signals win much
   more than 0.55–0.70 ones, raise `min_confidence` to 0.70 and trade
   only the best setups.
3. **Turn on the news filter.** Add a Finnhub key, set
   `news_filter_enabled=true`, and choose `news_block_minutes` (default
   30) around HIGH-importance USD events.
4. **Tune the shock meter.** `shock_multiple` (how violent a candle
   must be, default 3.0×) and `shock_cooldown_minutes` (default 30).
5. **Cut API costs in dead markets.** `ai_skip_max_spread_pct` skips
   the DeepSeek calls when the spread is too wide (0 = disabled).
6. **Replay before trusting a past signal.** `replay <timestamp>` shows
   exactly what the robot knew at that moment — no hindsight cheating.
7. **Check the A/B verdict before big config changes.** `ab-compare`
   proves whether a change really improves the robot vs the tagged
   LEGACY_BASELINE — never tune on vibes.
8. **Journaling is automatic now.** Outcomes, MFE/MAE and calibration
   are stored for you; you only need to review them.
9. **Planned upgrades that will raise accuracy further:**
   - Session-overlap weighting (prefer 13:30–16:30 UTC signals)
   - DXY from MT5 in real time (Windows machine)
   - Multi-symbol confirmation

---

*Guide updated after the INTELLIGENCE_V2 upgrade (all 57 spec sections
delivered). Current config: 15m entries, 4h bias, DXY gate on, session
gate ON (London/NY/Sydney, set SESSION_SYDNEY="" to drop Sydney), shock
detection ON, news filter OFF (opt-in), statistical quality OFF
(opt-in).*
