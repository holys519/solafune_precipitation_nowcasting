# Round 8 — sustained experiment campaign (2026-07-27 → ~07-31)

Directive: keep experimenting until we can make a proper judgment, roughly ≥2× the volume run so
far (exp001→064), until ~3 days before the deadline (~07-31; deadline ~08-03). Run many iterations
per promising axis (single-shot go/no-go was premature), diagnose the core bottleneck in parallel,
and **log every experiment here so none is wasted**.

## Defensible floor (never lose this)
`exp056_seed_ens_42_456` = **0.68277** public (green, causal-replay-clean). All new work is measured
against exp056 gate baseline **fold0 0.28159 / fold4 0.58503** and champion LB 0.68277.

## What is EXHAUSTED (do not re-run — n≫1, independently confirmed)
- Feature engineering: exp047/048/049/050/051/059 + MahmoudElshahed's 9 nulls. Dead.
- Overlap patch / successor(≥T) rows / non-causal smoothing: banned (red).
- Seed ensemble scale-up: 2-seed 42+456 best (0.68277); 4/6-seed worse. Plateaued.
- Pretrained backbone as a SOLO model: effv2s ties, none beats champion (capacity ≠ bottleneck).
- Crude channel-stack history: cr1 (no history, exp056) beats cr2/cr3/cr4 monotonically as more
  history channels are stacked. Confirmed n=3 (2026-07-27). Crude stacking is dead; only the
  "temporal architecture" reformulation (shared encoder + attention/ConvLSTM) is still open.

## Core bottleneck (working diagnosis)
Train→eval distribution shift over 20 non-overlapping train locations vs 18 eval. Adding capacity
(pretrained) or information (2h history, n=1 crude) both improved TRAIN fit but not held-out
generalization. OOF↛LB transfer only holds for architecture/simplification changes. So iterate on:
(a) things that change generalization behavior (regularization, diverse ensembling, output
factorization), not raw capacity/features; (b) better USE of history/architecture than the crude
first attempts.

## Verification discipline (every candidate before an LB slot)
1. fold0/4 gate vs 0.28159 / 0.58503 (both improve, or one improves + other within ~0.004 noise).
2. If gate passes → 5-fold OOF; architecture-type OOF is trustworthy (doc/oof_lb_transfer...).
3. `l_eda/exp005/submission_gate.py` GO + `scripts/verify_causal_replay.py` clean before submit.
4. super-site twin grouping for any honest-OOF (doc/twin_fold_and_trap_audit).

## Experiment ledger (update as each completes)

| exp | axis | hypothesis | config | fold0 | fold4 | verdict |
| --- | --- | --- | --- | ---: | ---: | --- |
| exp063 | history | 2h predecessor helps | cr=4, 207ch, stack | 0.28474 | 0.59410 | ✗ both worse (too much/crude) |
| exp064_effv2s | capacity | pretrained beats scratch | effv2s | 0.28208 | 0.58531 | ~tie (solo no; keep for diverse blend) |
| exp064_effb3 | capacity | " | effb3 | 0.28833 | 0.58135 | mixed (fold4 best-in-class) |
| exp064_resnet34 | capacity | " | resnet34 | 0.28699 | 0.59281 | ✗ worse |
| exp064_convnext | capacity | " | convnext, lr1e-3 | NaN | NaN | ✗ diverged (retry lr2e-4) |
| exp064_swin | capacity | " | swin, lr1e-3 | NaN | NaN | ✗ diverged (retry lr2e-4) |
| **Batch A (history design space)** | | | | | | |
| exp063_cr2 | history | a LITTLE history (T-30 only) is the sweet spot | cr=2, 105ch | 0.28302 | 0.59143 | ✗ fold4 outside noise (+0.0064); worse than cr1 |
| exp063_cr3 | history | T-30/T-60 | cr=3, 156ch | 0.28838 | 0.59237 | ✗ both worse, clearly |
| **Batch B (encoder NaN rescue)** | | | | | | |
| exp064_convnext_lr2e4 | capacity | convnext trains stably at lr2e-4 + plateau | convnext lr2e-4 | 0.28470 | 0.58687 | NaN fixed, but ties baseline (both within noise, neither improves) — no clean pass |
| exp064_swin_lr2e4 | capacity | swin stable at lr2e-4 | swin lr2e-4 | **0.27404** | 0.58572 | ✓ GATE PASS — fold0 improves by 0.0076, fold4 within noise (+0.0007). Best result since exp056. → next: 5-fold OOF |
| **Batch C (architecture-diverse ensemble)** | | | | | | |
| exp056 × effv2s | ensemble | different feature extractor decorrelates errors where seeds could not | equal blend | — | — | pending effv2s submit |

## Parallel diagnostic track
- D1: OOF residual anatomy by location/satellite/amount-regime on exp056 (where does it fail?) →
  targets the next architecture iteration. (uses existing analyze_oof outputs; super-site grouped.)
- D2: is the honest ceiling ~0.68 (information-limited) or is 0.63-0.65 reachable? Track vs
  MahmoudElshahed's 0.6798 defensible + his 0.63-0.65 prediction.

## Batch D (2026-07-28) — complete gate-passers to submittable + regularization sweep
Launched after Batch A/B sat idle ~20h unreviewed; results pulled from logs and folded in above.
| exp | axis | hypothesis | jobs | status |
| --- | --- | --- | --- | --- |
| exp064_swin_lr2e4 | complete-to-submit | fold0/4 gate-passed; folds1-3 needed for honest 5-fold OOF + submission | 3952674-76 | running |
| exp064_convnext_lr2e4 | complete-to-submit | NaN-fixed tie; still useful as a diverse-blend member | 3952677-79 | queued |
| exp056_wd5x (weight_decay 5e-4) | regularization | targets diagnosed generalization/distribution-shift bottleneck directly | 3952681-82 (fold0/4) | queued |
| exp056_wd10x (weight_decay 1e-3) | regularization | " (larger step) | 3952683-84 (fold0/4) | queued |

## Submission-ready deliverables (2026-07-28 snapshot)
Already-trained (5-fold) zips sitting unreviewed since 07-27, plus one new diverse blend built today:
1. `exp064_effv2s_submission.zip` — solo pretrained backbone, ties champion (fold0 0.28208/fold4 0.58531)
2. `exp064_effb3_submission.zip` — solo, mixed (fold4 best-in-class 0.58135, fold0 worse 0.28833)
3. `exp056_seed_ens_4best_submission.zip` — 4-seed variance-reduction ensemble (built 07-27, never scored)
4. `exp056_seed_ens_6_submission.zip` — 6-seed variance-reduction ensemble (built 07-27, never scored)
5. `diverse_champion_x_effv2s_submission.zip` — **NEW** equal blend of champion 2-seed ensemble x effv2s (architecture-diverse, Batch C's actual idea), built 2026-07-28
Not yet run through `l_eda/exp005/submission_gate.py` (needs OOF, not just fold0/4/eval-only) — do
that before spending an actual LB submission slot on any of these.

## Next generations (queued ideas, launch as batches free up)
- Temporal architecture for history (shared per-frame encoder + temporal attention/ConvLSTM) —
  proper use of history vs crude channel-stack.
- Regularization sweep on exp056 (dropout/weight-decay/augment strength) — directly targets the
  generalization bottleneck the diagnosis points to.
- exp056 factorization internals (decoder capacity, loss weights, amount-head design).
- Diverse blend generations as new architecture-distinct members clear the gate.
