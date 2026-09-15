# How Your Trading Robot Thinks — User-Friendly Guide

A step-by-step explanation of the XAUUSD robot's decision process, written
to help you judge its signals and improve its accuracy.

---

## 1. The Big Picture

```
 MARKET DATA         3 DEEPSEEK BRAINS         THE JUDGE             YOU
 (every 15 min)      (research & vote)     (hard safety gates)   (decide)
┌─────────────┐    ┌───────────────────┐   ┌─────────────────┐   ┌───────┐
│ 15m candles │───▶│ 1. Technical      │──▶│ Kill-switch     │──▶│Telegram
│ 4h candles  │    │ 2. Sentiment      │   │ Neutral/confid. │   │signal │
│ DXY gauge   │    │ 3. Regime         │   │ DXY gate        │   │  ↓    │
│ indicators  │    └───────────────────┘   │ HTF bias gate   │   │ MT5   │
└─────────────┘          │ weighted vote    │ Position limits │   │ (you) │
                         ▼                  │ Sizing/SL/TP    │   └───────┘
                    side + confidence ─────▶│ proposal        │
                                            └─────────────────┘
```

The key design rule: **the AI proposes, the machine disposes.**
DeepSeek never decides anything final — the deterministic risk engine
holds the final authority and cannot be overruled.

---

## 2. Step 1 — The Eyes: Gathering Data (every new 15m candle)

| Data | Source | Notes |
|---|---|---|
| XAUUSD 15m candles | yfinance gold futures → PAXG/USD fallback (24/7) | Last 300 candles |
| XAUUSD 4h candles | same sources | For the trend bias |
| DXY gauge (0–100) | MT5 `DXY_U6` → yfinance `DX-Y.NYB` fallback | 100 = weak dollar = bullish gold |
| Indicator snapshot | computed locally (pandas) | EMA 20/50/200, RSI, MACD, ATR, ADX, Bollinger, volume |

All indicators are computed **deterministically by our code** — the AI
never calculates math, it only reads the results.

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

---

## 4. Step 3 — The Vote: Fusion

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

---

## 5. Step 4 — The Judge: Risk Engine Hard Gates (in order)

If ANY gate fails, the signal is refused with a written reason:

| # | Gate | Rule |
|---|---|---|
| 1 | Kill-switch | If halted (daily loss −3% / drawdown −10% / equity 0), everything is refused |
| 2 | Neutral | Fused signal must have a direction |
| 3 | Confidence | `confidence ≥ 0.55` |
| 4 | **DXY concurrency** | LONG needs gauge ≥ 55 (weak dollar); SHORT needs gauge ≤ 45 (strong dollar) |
| 5 | **HTF bias** | LONG only in 4h uptrend; SHORT only in 4h downtrend; choppy 4h (ADX < 20) blocks everything |
| 6 | Duplicates | One open position per symbol; max 3 total |
| 7 | Validity | Price and ATR must be sane |
| 8 | Sizing | Risk = 1% of equity; size = risk ÷ stop distance; exposure capped at 25% of equity; dust (< $5 notional) refused |

**SL & TP are mechanical:** stop = entry ∓ 2×ATR, target = entry ± 4×ATR
(always a 2:1 reward/risk).

---

## 6. Step 5 — The Output

A passing signal becomes a **proposal**:
- saved to the database (with full audit log)
- sent to your phone: entry, stop, target, size, confidence, the DXY
  line and the 4h bias line
- printed in the console with an `approve` command

**Nothing is executed automatically — you remain the decision-maker.**

---

## 7. Simulation A — A Signal That Passes

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
4. DXY: 62 ≥ 55 ✓ (weak dollar = bullish gold)
5. HTF 4h bias: bull ✓
6. No open position on XAUUSD ✓
7. Price/ATR valid ✓
8. Sizing: risk = 100 USD (1% of 10k) → stop distance 10 → size 10 units;
   exposure cap 25% of equity = 2,500 → shrunk to 2,500 ÷ 2,400 ≈ 1.04 units ✓
   (the position shrinks, so the real money at risk drops to ~1.04 × 10 ≈ 10.40 USD)

**Proposal:** BUY 1.04 @ 2400, SL 2390, TP 2420, confidence 0.56
(planned risk 100 USD — but the 25% exposure cap actually shrinks the position)
→ Telegram message arrives with all details. **You decide.**

---

## 8. Simulation B — Rejected by the DXY Gate

Same agents say LONG with confidence 0.70 — but the gauge is 48.
The dollar is not weak enough. The judge refuses:

> "DXY concurrency: gauge 48 not weak-dollar (need >= 55) for a LONG"

The robot stays flat. This is the gate doing its job.

## 9. Simulation C — Rejected by the HTF Bias Gate

Agents say SHORT, confidence 0.65, DXY gauge 30 (strong dollar ✓) —
but the 4h trend is bullish (EMA 50 above EMA 200).
Fighting the 4h trend is a losing trade statistically, so:

> "HTF bias: bull on 4h blocks SHORT"

Refused. "The trend is your friend" — enforced by machine, not by hope.

---

## 10. The Loop Over Time (a typical day, 24/7 test mode)

| Moment | What happens |
|---|---|
| Every 60s | Tick: fetch latest data, manage open positions (stop/target) 24/7 |
| Every new 15m candle | Full analysis cycle (3 DeepSeek calls + gates) |
| Candle with no new info | No analysis — no wasted cost |
| Kill-switch trip | Alert on Telegram, no new proposals |
| Every day (UTC) | DeepSeek balance check → alert if < 1 USD |

---

## 11. Honest Limits of the Current "Research"

- ❌ No economic calendar / news filter yet (V2 roadmap)
- ❌ No SMC structure detection yet (BOS/CHoCH/FVG/order blocks — V2)
- ❌ No multi-symbol confirmation (e.g., GBPUSD) yet
- ⚠️ Weekends: DXY gauge frozen at Friday's close
- ⚠️ Degraded mode: if DeepSeek is unreachable, heuristics take over
  (you are alerted — treat those signals with extra caution)

---

## 12. How to Improve Decision Accuracy (your levers)

1. **Read the rejections.** Every refusal tells you which gate fired.
   Watch the distribution: too many "confidence below minimum" → lower
   `MIN_CONFIDENCE`; too many DXY rejections → check `DXY_LONG_MIN`/`DXY_SHORT_MAX`.
2. **Track confidence vs outcome.** After 2–4 weeks, compare: did signals
   with confidence ≥ 0.70 win more than those at 0.55–0.70? If yes, raise
   `MIN_CONFIDENCE` to 0.70 and you trade only the best setups.
3. **Review the technical agent's notes.** They contain the reasoning.
   If the reasoning looks wrong repeatedly, the weights can be tuned
   (`WEIGHT_TECHNICAL`, `WEIGHT_REGIME`, `WEIGHT_SENTIMENT`).
4. **Keep a simple journal:** date, signal, confidence, gates, outcome.
   This data is gold for tuning.
5. **Planned upgrades that will raise accuracy:**
   - News filter (avoid NFP/CPI moments)
   - SMC detection (BOS/CHoCH/FVG/order blocks)
   - Session-overlap weighting (prefer 13:30–16:30 UTC signals)
   - DXY from MT5 in real time (Windows machine)

---

*Report generated for testing-phase tuning. Current config: 15m entries,
4h bias, DXY gate on, session gate OFF (24/7 testing).*
