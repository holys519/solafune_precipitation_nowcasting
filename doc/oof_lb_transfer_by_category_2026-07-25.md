# OOF→LB transfer, stratified by change type (2026-07-25)

Meta-analysis of every green submission this session for which we have BOTH a 5-fold pooled OOF
(`outputs/analysis/<exp>/analysis_summary.json` `oof_official_metric`) and a realized public LB.
Motivated by three consecutive OOF/LB inversions (exp040_metric, exp047_sigmafixed, exp055 blend)
that nearly cost us submissions on models that looked like OOF improvements.

## The pairs

| Experiment | Change type | pooled OOF | Public LB | vs sigmafixed OOF | vs sigmafixed LB |
| --- | --- | ---: | ---: | ---: | ---: |
| exp038 | baseline | 0.61313 | 0.68916 | +0.00485 | +0.00252 |
| exp038_sigmafixed | simplification (drop predicted-σ head) | 0.60828 | 0.68664 | — | — |
| exp050_sigmafixed | feature (split-window BTD) + sigma_fix | 0.60643 | 0.68702 | −0.00185 | +0.00038 |
| exp040_metric | loss-shaping (tile-RMSE metric loss) | 0.60682 | 0.69552 | −0.00146 | **+0.00888** |
| exp047_sigmafixed | feature + geography (solar/hemisphere) | 0.60747 | 0.70014 | −0.00081 | **+0.01350** |
| **exp056** | **architecture (mean×shape factorization)** | **0.60252** | **0.68396** | **−0.00576** | **−0.00268** |

## The pattern (this is the actionable finding)

**Architecture-level and simplification changes transfer OOF→LB faithfully; feature additions and
loss-shaping tricks invert or wash out.**

- **exp056 (architecture)**: biggest OOF gain AND best LB. The project's OOF→LB regression
  (`LB ≈ 1.268×OOF − 0.080`, `l_eda/exp003`) predicts 0.6839 for its OOF 0.60252 — the realized
  0.68396 matches almost exactly (residual +0.0001).
- **exp038_sigmafixed (simplification)**: also a clean, faithful transfer (regression predicts
  0.6913, realized 0.68664 — even slightly better than predicted).
- **exp047_sigmafixed (feature + geography)**: regression predicts 0.6903, realized **0.70014** —
  a +0.0098 positive residual (much worse than even the regression's estimate). The
  hemisphere × satellite-one-hot feature let the model memorize per-region train climate; this is
  a domain-shift trap that OOF (train-locations only) structurally cannot see.
- **exp040_metric (loss-shaping)** and **exp055 (OOF-optimal blend)**: both inverted despite better
  OOF (see `doc/public_scores.md`).
- **exp050_sigmafixed (feature)**: OOF −0.00185 but LB +0.00038 — a wash, not the improvement OOF
  advertised.

## Operating rules derived from this (apply to the endgame)

1. **Trust an OOF gain in proportion to how structural the change is.**
   - Architecture / simplification change → OOF gain is credible; the OOF→LB regression is
     well-calibrated for these.
   - Feature addition / loss-shaping / blend-weight fit → treat the OOF gain as **not yet real**.
     Require a live LB read before adopting, and never spend a final-submission slot on one purely
     on OOF.
2. **Any feature touching location/geography is guilty until proven innocent** (exp047's +0.0135).
   The 20 non-overlapping train locations make climate-memorization invisible to OOF.
3. **For the two experiments now gating**: exp060 (architecture: exact total×distribution) — if it
   improves OOF, that gain is credible by rule 1. exp059 (feature: NODATA emissive masking) — even
   a positive OOF result must be treated skeptically and confirmed on LB before adoption.
4. This complements, and is stricter than, `l_eda/exp005/submission_gate.py`: the gate checks
   whether the OOF gain is statistically real *within train locations*; this note says even a real
   within-train gain does not transfer for feature/loss changes.

## Champion status unchanged

exp056 (LB 0.68396) remains the green champion. Its OOF lead is the one OOF number this session we
can most trust, precisely because it comes from an architecture change.
