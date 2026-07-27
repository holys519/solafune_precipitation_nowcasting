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
| exp063_cr2 | history | a LITTLE history (T-30 only) is the sweet spot | cr=2, 105ch | — | — | running |
| exp063_cr3 | history | T-30/T-60 | cr=3, 156ch | — | — | running |
| **Batch B (encoder NaN rescue)** | | | | | | |
| exp064_convnext_lr2e4 | capacity | convnext trains stably at lr2e-4 + plateau | convnext lr2e-4 | — | — | running |
| exp064_swin_lr2e4 | capacity | swin stable at lr2e-4 | swin lr2e-4 | — | — | running |
| **Batch C (architecture-diverse ensemble)** | | | | | | |
| exp056 × effv2s | ensemble | different feature extractor decorrelates errors where seeds could not | equal blend | — | — | pending effv2s submit |

## Parallel diagnostic track
- D1: OOF residual anatomy by location/satellite/amount-regime on exp056 (where does it fail?) →
  targets the next architecture iteration. (uses existing analyze_oof outputs; super-site grouped.)
- D2: is the honest ceiling ~0.68 (information-limited) or is 0.63-0.65 reachable? Track vs
  MahmoudElshahed's 0.6798 defensible + his 0.63-0.65 prediction.

## Next generations (queued ideas, launch as batches free up)
- Temporal architecture for history (shared per-frame encoder + temporal attention/ConvLSTM) —
  proper use of history vs crude channel-stack.
- Regularization sweep on exp056 (dropout/weight-decay/augment strength) — directly targets the
  generalization bottleneck the diagnosis points to.
- exp056 factorization internals (decoder capacity, loss weights, amount-head design).
- Diverse blend generations as new architecture-distinct members clear the gate.
