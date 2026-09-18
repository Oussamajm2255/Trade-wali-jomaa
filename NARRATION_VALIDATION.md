# NARRATION TELEGRAM — VALIDATION AVANT DÉPLOIEMENT

Livrable Partie 2 (couche de présentation). Document en français : c'est
l'artefact de validation destiné à l'humain avant mise en production.

**Règle cardinale (jamais violée)** : la narration est une reformulation
de la décision déterministe déjà prise — elle n'a ni le droit d'inventer
une donnée, ni celui de changer le verdict, ni celui de sonner plus
confiante que le score réel. Elle est une fonction pure du signal_record.

---

## 1. Ce qui a été construit

| Pièce | Fichier | Rôle |
|---|---|---|
| Moteur narratif | `trading_agent/notify/narration.py` | Texte français déterministe, fonction pure du record |
| Câblage | `trading_agent/notify/telegram.py` | Narration en haut, `── Détails (audit) ──` en dessous — rien n'est supprimé |
| Config | `telegram_narrative_enabled: bool = True` | Désactivation possible sans toucher au pipeline |
| Tests | `tests/test_narration.py` (22 tests) | Ton proportionnel, aucune donnée fabriquée, structure |

### Garanties mécaniques (vérifiées par les tests)

1. **Ton ≤ score réel.** Trois paliers sur `raw_confidence` (ou le palier
   statistique `fusion.tier` s'il est stocké) : LOW < 0.55 → « pas de
   conviction claire » / MEDIUM → « conviction mesurée » / HIGH ≥ 0.75 →
   « conviction nette ». Un raw de 0.12 produit une vraie incertitude.
2. **UNVALIDATED dit à voix haute.** Si `raw_confidence` existe mais que
   `calibrated_confidence` est `null`, la narration ajoute :
   « Ma calibration n'est pas encore validée statistiquement — ces
   scores restent indicatifs, pas des probabilités. »
3. **Aucun chiffre fabriqué.** Un test extrait tous les nombres du texte
   et vérifie qu'ils proviennent du record (liquidité, VWAP, stop,
   objectif). Aucun niveau n'est halluciné — si rien n'est calculé, le
   texte le dit : « Aucun niveau calculé — observation du prix uniquement. »
4. **Résumé, jamais un second avis.** La narration ne connaît que le
   record : elle ne peut ni ajouter une nuance, ni suggérer, ni biaiser.
5. **Les 21 raisons `NoTradeReason` sont couvertes** (phrases + condition
   de ré-entrée « À surveiller »), y compris l'alias hérité `DXY_FILTER`.

---

## 2. Avant / après sur 3 cas réels (DB existantes)

### CAS 1 — Rejet LOW_CONFIDENCE (record LIVE, `trading.db`)

`XAUUSD-20260916-000001`, raw = 0.259, calibré = null, conflits CONFLICTED.
C'est la décision réelle prise par le robot live le 2026-09-16.

**AVANT — dump technique :**

```
🚫 SIGNAL REJETÉ XAUUSD (15m)
Raison : confidence 0.26 below minimum 0.55 (LOW_CONFIDENCE)

Confiance brute : 0.26
Biais MTF : 1D bear · 4H neutre · 1H neutre · 15m neutre
Régime : range | DXY : Bearish (USD strong) | Tendance DXY : haussière | Session : Asie
Structure : BOS, FVG, balayage de liquidité
Soutient : structure de marché fort (0.90) · structure: BOS, FVG, balayage de liquidité · DXY: Bearish (USD strong)
Contredit : contexte DXY faible (0.40) · volatilité faible (0.35) · gauge 48 not weak-dollar for a LONG · price below all known supports opposes a LONG
Gates : market_data ✓ | data_quality ✓ | cost_control ✓ | agents ✓ | confidence ✗
DXY : long (0.55)
  ↳ DXY sits at 99.644 with a bullish trend (ADX 32.77)...
REGIME : flat (0.72)
  ↳ The deterministic entry-timeframe regime is range with ADX at 19.76...
TECHNICAL : long (0.42)
  ↳ Price broke above the 4351.30 swing high with a fresh bullish BOS...
Qualité des données : degraded
Version : LEGACY_BASELINE
```

**APRÈS — narration + détails :**

```
🚫 SIGNAL REJETÉ XAUUSD (15m)

Un balayage de liquidité vient de se produire, le marché est en range, le dollar est fort.
Je n'ai pas de conviction claire — les votes sont trop faibles ou trop partagés pour
engager du risque. Je reste à l'écart.
La structure penche pour une entrée, mais le contexte DXY et la volatilité retiennent
le déclencheur — la tension est réelle.
À surveiller : un signal plus net émerge (BOS ou CHoCH confirmé). Aucun niveau calculé —
observation du prix uniquement.
Ma calibration n'est pas encore validée statistiquement — ces scores restent indicatifs,
pas des probabilités.

── Détails (audit) ──
Raison : confidence 0.26 below minimum 0.55 (LOW_CONFIDENCE)

Confiance brute : 0.26
Biais MTF : 1D bear · 4H neutre · 1H neutre · 15m neutre
Régime : range | DXY : Bearish (USD strong) | Tendance DXY : haussière | Session : Asie
Structure : BOS, FVG, balayage de liquidité
Soutient : structure de marché fort (0.90) · structure: BOS, FVG, balayage de liquidité · DXY: Bearish (USD strong)
Contredit : contexte DXY faible (0.40) · volatilité faible (0.35) · gauge 48 not weak-dollar for a LONG · price below all known supports opposes a LONG
Gates : market_data ✓ | data_quality ✓ | cost_control ✓ | agents ✓ | confidence ✗
DXY : long (0.55)
  ↳ DXY sits at 99.644 with a bullish trend (ADX 32.77)...
REGIME : flat (0.72)
  ↳ The deterministic entry-timeframe regime is range with ADX at 19.76...
TECHNICAL : long (0.42)
  ↳ Price broke above the 4351.30 swing high with a fresh bullish BOS...
Qualité des données : degraded
Version : LEGACY_BASELINE
```

Ton vérifié : raw 0.259 → palier LOW → « pas de conviction claire », aucun
langage assuré. La calibration absente est dite explicitement.

---

### CAS 2 — Rejet CONFLICTED (record réel, `backtest_intelligence_v2.db`)

`XAUUSD-20260826-000063`, raw = 0.557, raison MTF_CONFLICT (2 conflits
déterministes : 4h bull vs short 15m ; gauge DXY 48 pour un SHORT).

**AVANT — dump technique :**

```
🚫 SIGNAL REJETÉ XAUUSD (15m)
Raison : deterministic conflicts on 2 axis(es) (dxy, mtf): 4h bias bull opposes short;
gauge 48 not strong-dollar for a SHORT (MTF_CONFLICT)

Confiance brute : 0.56
Biais MTF : 1D neutre · 4H bull · 1H neutre · 15m bear
Régime : tendance baissière | DXY : Bearish (USD strong) | Tendance DXY : haussière | Session : Chevauchement Londres+NY
Structure : BOS, FVG, balayage de liquidité
Soutient : régime fort (1.00) · ratio risque/rendement fort (1.00) · structure de marché fort (0.90) · structure: BOS, FVG, balayage de liquidité · DXY: Bearish (USD strong)
Contredit : alignement MTF faible (0.15) · volatilité faible (0.35) · 4h bias bull opposes short · gauge 48 not strong-dollar for a SHORT
Gates : market_data ✓ | data_quality ✓ | cost_control ✓ | agents ✓ | shock ✗ | no_trade ✗
DXY : short (0.92) [heuristique]
  ↳ Deterministic DXY context score=27.0 (Bearish (USD strong))...
REGIME : down (0.57) [heuristique]
  ↳ Deterministic regime engine: trend_down (ADX=28.67, ATR ratio=1.4113).
TECHNICAL : short (0.59) [heuristique]
  ↳ Deterministic fallback: EMA20/50/200 alignment...
Qualité des données : pass
Version : LEGACY_BASELINE
```

**APRÈS — narration + détails :**

```
🚫 SIGNAL REJETÉ XAUUSD (15m)

Un balayage de liquidité vient de se produire, la tendance est baissière, le dollar est fort.
Les signaux se contredisent nettement — les unités de temps se contredisent. Je reste à l'écart.
La structure, le régime et le ratio risque/rendement penchent pour une entrée, mais
l'alignement MTF et la volatilité retiennent le déclencheur — la tension est réelle.
À surveiller : les unités de temps se réalignent dans le même sens. Aucun niveau calculé —
observation du prix uniquement.
Ma calibration n'est pas encore validée statistiquement — ces scores restent indicatifs,
pas des probabilités.

── Détails (audit) ──
Raison : deterministic conflicts on 2 axis(es) (dxy, mtf): 4h bias bull opposes short;
gauge 48 not strong-dollar for a SHORT (MTF_CONFLICT)

Confiance brute : 0.56
Biais MTF : 1D neutre · 4H bull · 1H neutre · 15m bear
Régime : tendance baissière | DXY : Bearish (USD strong) | Tendance DXY : haussière | Session : Chevauchement Londres+NY
Structure : BOS, FVG, balayage de liquidité
Soutient : régime fort (1.00) · ratio risque/rendement fort (1.00) · structure de marché fort (0.90) · structure: BOS, FVG, balayage de liquidité · DXY: Bearish (USD strong)
Contredit : alignement MTF faible (0.15) · volatilité faible (0.35) · 4h bias bull opposes short · gauge 48 not strong-dollar for a SHORT
Gates : market_data ✓ | data_quality ✓ | cost_control ✓ | agents ✓ | shock ✗ | no_trade ✗
DXY : short (0.92) [heuristique]
  ↳ Deterministic DXY context score=27.0 (Bearish (USD strong))...
REGIME : down (0.57) [heuristique]
  ↳ Deterministic regime engine: trend_down (ADX=28.67, ATR ratio=1.4113).
TECHNICAL : short (0.59) [heuristique]
  ↳ Deterministic fallback: EMA20/50/200 alignment...
Qualité des données : pass
Version : LEGACY_BASELINE
```

Le « pourquoi » est fusionné en une seule logique racontée (la tension
soutient/freins), pas deux listes. La condition de reconsidération
(« les unités de temps se réalignent ») vient des données réelles.

---

### CAS 3 — Proposition (record réel, `backtest_intelligence_v2.db`)

`XAUUSD-20260821-000041`, raw = 0.767, qualité setup 0.75, ALIGNED.

**Note d'honnêteté obligatoire** : il n'existe **aucun** enregistrement A+
dans aucune DB du projet (trading.db, backtest_*.db). A+ exige le palier
statistique HIGH, qui exige une calibration (`tier_high_min_calibrated`),
et la calibration n'a jamais existé : `calibrated_confidence` est `null`
sur 100 % des records. Le système est UNVALIDATED — prétendre produire un
exemple A+ serait fabriquer une donnée. Le cas 3 est donc la meilleure
proposition réelle jamais produite, présentée telle quelle ; la narration
l'étiquette correctement comme non calibrée.

**AVANT — dump technique :**

```
🎯 SIGNAL XAUUSD — LONG

Entrée : 4,635.70
Stop : 4,615.73
Objectif : 4,675.63
RR : 2.00 | Taille : 0.5400 | Risque : 100.14 $

Confiance brute : 0.77
Qualité du setup : 0.75 (conflits : ALIGNED)
Biais MTF : 1D neutre · 4H bull · 1H bull · 15m bull
Régime : tendance haussière | DXY : Bullish (USD weak) | Tendance DXY : baissière | Session : Londres
Structure : FVG, balayage de liquidité
Soutient : alignement MTF fort (1.00) · régime fort (1.00) · structure de marché fort (0.70) · volatilité fort (0.90) · structure: FVG, balayage de liquidité · DXY: Bullish (USD weak)
Contredit : ratio risque/rendement faible (0.30)
Invalidation : setup invalidé si le prix clôture au-delà du SL (4,615.7331)
Gates : market_data ✓ | data_quality ✓ | cost_control ✓ | agents ✓ | shock ✗ | statistical_quality ✓ | dxy_concurrency ✓ | htf_bias ✓ | positions ✓ | final ✓
DXY : long (0.80) [heuristique]
REGIME : up (0.96) [heuristique]
TECHNICAL : long (0.78) [heuristique]

Version : LEGACY_BASELINE

⏳ EN ATTENTE D'APPROBATION HUMAINE
✅ Approuver : python -m trading_agent.main approve pid-x
❌ Rejeter  : python -m trading_agent.main reject pid-x
```

**APRÈS — narration + détails :**

```
🎯 SIGNAL XAUUSD — LONG

Un balayage de liquidité vient de se produire, la tendance est haussière, le dollar est
faible — le contexte est net. Je propose l'achat avec conviction.
L'alignement MTF, la structure et le régime penchent pour l'achat, mais le ratio
risque/rendement retient le déclencheur — la tension est réelle.
À surveiller : l'objectif à 4,675.63 ; 4,615.73 invalide le setup.
Ma calibration n'est pas encore validée statistiquement — ces scores restent indicatifs,
pas des probabilités.

── Détails (audit) ──

Entrée : 4,635.70
Stop : 4,615.73
Objectif : 4,675.63
RR : 2.00 | Taille : 0.5400 | Risque : 100.14 $

Confiance brute : 0.77
Qualité du setup : 0.75 (conflits : ALIGNED)
Biais MTF : 1D neutre · 4H bull · 1H bull · 15m bull
Régime : tendance haussière | DXY : Bullish (USD weak) | Tendance DXY : baissière | Session : Londres
Structure : FVG, balayage de liquidité
Soutient : alignement MTF fort (1.00) · régime fort (1.00) · structure de marché fort (0.70) · volatilité fort (0.90) · structure: FVG, balayage de liquidité · DXY: Bullish (USD weak)
Contredit : ratio risque/rendement faible (0.30)
Invalidation : setup invalidé si le prix clôture au-delà du SL (4,615.7331)
Gates : market_data ✓ | data_quality ✓ | cost_control ✓ | agents ✓ | shock ✗ | statistical_quality ✓ | dxy_concurrency ✓ | htf_bias ✓ | positions ✓ | final ✓
DXY : long (0.80) [heuristique]
REGIME : up (0.96) [heuristique]
TECHNICAL : long (0.78) [heuristique]

Version : LEGACY_BASELINE

⏳ EN ATTENTE D'APPROBATION HUMAINE
✅ Approuver : python -m trading_agent.main approve pid-x
❌ Rejeter  : python -m trading_agent.main reject pid-x
```

Le niveau à surveiller (objectif 4 675,63 et invalidation 4 615,73) vient
du record, jamais d'un calcul nouveau. La phrase de hedge reste présente
car ce signal est, lui aussi, statistiquement non validé.

---

## 3. Garde-fous vérifiés par les tests (22 tests, tous verts)

| Garde-fou | Test |
|---|---|
| Ton proportionnel par palier (LOW / MEDIUM / HIGH) | `test_low/medium/high_tone_*` |
| Rejet LOW_CONFIDENCE humble, rejet CONFLICTED nomme la tension | `test_low_confidence_rejection_stays_humble`, `test_conflicted_rejection_names_the_tension` |
| Les 21 `NoTradeReason` ont une narration | `test_every_no_trade_reason_has_a_narrative` |
| Aucun chiffre fabriqué (audit des nombres vs record) | `test_no_fabricated_numbers` |
| Niveau à surveiller issu de la liquidité réelle | `test_watch_level_comes_from_real_liquidity` |
| VWAP volume tick étiqueté (honnêteté §7) | `test_tick_volume_vwap_is_labelled_in_narrative` |
| Hedge UNVALIDATED présent si calibré null, absent sinon | `test_low_tone_never_sounds_confident`, `test_medium_tone_is_measured` |
| Déterminisme (mêmes entrées → même texte) | `test_narrative_is_deterministic`, `test_full_message_is_deterministic` |
| Détails techniques intacts sous le séparateur | `test_proposal_message_narrative_above_details`, `test_rejection_message_narrative_above_details` |
| Refus pré-AI (NEWS_RISK) : narration courte, rien d'inventé | `test_pre_ai_refusal_narrative_stays_concise` |
| Désactivation par config sans perte | `test_narrative_can_be_disabled` |

Suite complète existante (tests/test_telegram.py, 31 tests) : **toutes
vertes sans modification** — la couche de présentation n'a rien cassé.

---

## 4. Statut de déploiement

- La narration est **activée par défaut** (`telegram_narrative_enabled=true`)
  et n'exige aucune API supplémentaire : c'est du templating déterministe,
  zéro coût, zéro latence, zéro risque de fabulation LLM.
- Désactivable en une variable d'environnement si besoin.
- Rien dans la chaîne de décision n'a été modifié : `fusion/engine.py` et
  `risk/engine.py` sont intacts, la narration tourne APRÈS eux.
