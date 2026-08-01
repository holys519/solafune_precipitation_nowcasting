# Round 8 — sustained experiment campaign (2026-07-27 → ~07-31)

Directive: keep experimenting until we can make a proper judgment, roughly ≥2× the volume run so
far (exp001→064), until ~3 days before the deadline (~07-31; deadline ~08-03). Run many iterations
per promising axis (single-shot go/no-go was premature), diagnose the core bottleneck in parallel,
and **log every experiment here so none is wasted**.

## Defensible floor (never lose this)
`exp065_champion_ensemble_causal.zip` = **0.66966** public (green, 5-fold ensemble), scored
2026-07-30 16:03:44 — beats the effb3-solo floor (0.67201) by **-0.00235**, and beats the original
07-27 floor (`exp056_seed_ens_42_456`, 0.68277) by **-0.01311** overall. This is the architecture-
diverse 4-way ensemble (exp056/effb3/effv2s/swin_lr2e4) built by Track A (see Batch E below) —
`g_eda/exp011/nested_blend.py`'s honest outer-cross-fit OOF gain was -0.01191 vs best solo, gate
verdict GO, and it transferred (at ~20%) to a real, gate-confirmed LB win, not an inversion. Also
worth noting: `exp064_swin_lr2e4_submission.zip` solo = **0.67044** (2nd-best solo, -0.00157 vs
effb3), scored the same run. All new work should now be measured against the exp065 ensemble floor
0.66966.

`exp064_effb3_submission.zip` = 0.67201 (solo, 2026-07-30 07:16:38) — beats the previous floor
(`exp056_seed_ens_42_456`, 0.68277) by -0.01076, far outside this project's noise band.
`exp064_effv2s_submission.zip` = 0.67400 (3rd-best solo, -0.00877 vs the 07-27 floor). **This
overturned the "pretrained capacity ≠ bottleneck" EXHAUSTED item below** — see 2026-07-30 update.

## 2026-07-30 update: fold0/4 gate missed a major win — re-open pretrained-backbone track
`exp064_effb3` was fold0/4-gated as "mixed" (fold4 good 0.58135, fold0 worse 0.28833) and
`exp064_effv2s` as a tie (0.28208/0.58531 vs 0.28159/0.58503) — neither looked like a clear gate
pass. Both were nonetheless run to full 5-fold and submitted (as leftover deliverables from the
07-28 snapshot), and both crushed the champion on public LB. **Conclusion: the 2-fold (0/4) gate is
not reliable for pretrained-backbone changes** — it may be underestimating how much these
architectures help on the actual non-overlapping eval-location generalization problem, which is
exactly the axis the fold0/4 gate is supposed to test. Until this is understood, treat any
pretrained-backbone candidate as worth running to full 5-fold even on a "mixed"/"tie" gate result,
rather than discarding it. Priority follow-ups: (a) other backbones (resnet34/convnext/swin) already
trained under exp064 — re-score their actual 5-fold LB rather than trusting fold0/4 verdicts; (b)
ensemble the new effb3 champion with the old seed-ensemble champion (architecture-diverse blend,
same idea as Batch C but with the real champion now); (c) investigate *why* fold0/4 underestimated
this — do those two locations happen to be adversarial for pretrained backbones specifically?

## 2026-07-30 update #2: gate-reliability finding refined; Track A ensemble confirmed a real win
`exp064_swin_lr2e4` is the missing data point that refines the above: it was the one exp064 arm
with a CLEAN fold0/4 gate PASS (not mixed/tie: f0 0.27404 improved by 0.0076, f4 within noise) —
and it also delivered on LB (0.67044 solo, beating effb3). So the gate is not *globally*
unreliable for this axis; the specific failure mode is that a **"mixed"/"tie" verdict is not
evidence of "no value"** on the pretrained-backbone axis (effb3/effv2s proved that), while a
**clean pass still is a real, trustworthy positive signal** (swin_lr2e4 confirms this). Treat
these as two different categories of evidence going forward, not one blanket "don't trust the
gate."

Track A (see Batch E) then built `exp065_champion_ensemble_causal` — an architecture-diverse
4-way ensemble of exp056/effb3/effv2s/swin_lr2e4, weighted via `g_eda/exp011/nested_blend.py`'s
newly-extended (N=4) outer-cross-fit search — and it is now the **new overall champion (0.66966)**,
confirmed via `l_eda/exp005/submission_gate.py`'s GO verdict before submission (not after). The
honest nested OOF gain (-0.01191) transferred to LB at ~20% (-0.00235 realized) — lower than this
project's historical ~47-51% transfer rate for other architecture changes, but positive, gate-
confirmed, and NOT an inversion like the `exp055` postmortem. This validates the nested-CV-first
discipline this session added to the ensembling pipeline.

## What is EXHAUSTED (do not re-run — n≫1, independently confirmed)
- Feature engineering: exp047/048/049/050/051/059 + MahmoudElshahed's 9 nulls. Dead.
- Overlap patch / successor(≥T) rows / non-causal smoothing: banned (red).
- Seed ensemble scale-up: 2-seed 42+456 best among seed-ensemble variants (0.68277); 4/6-seed
  worse. Plateaued -- but superseded as overall champion by exp064_effb3 (0.67201), see above.
- ~~Pretrained backbone as a SOLO model: effv2s ties, none beats champion (capacity ≠
  bottleneck).~~ **RETRACTED 2026-07-30 — false. effb3 and effv2s both beat champion by a wide
  margin on full 5-fold LB. The fold0/4 gate that produced this conclusion was misleading for this
  axis. Capacity/pretrained backbones are reopened as the leading track.**
- Crude channel-stack history: cr1 (no history, exp056) beats cr2/cr3/cr4 monotonically as more
  history channels are stacked. Confirmed n=3 (2026-07-27). Crude stacking is dead; only the
  "temporal architecture" reformulation (shared encoder + attention/ConvLSTM) is still open.

## Core bottleneck (working diagnosis) — PARTIALLY OVERTURNED 2026-07-30
Previous diagnosis: train→eval distribution shift over 20 non-overlapping train locations vs 18
eval; adding capacity (pretrained) or information (2h history) improved TRAIN fit but not held-out
generalization, based on fold0/4 gate results. **This is now contradicted for pretrained backbones**
— effb3/effv2s solo pretrained models are the two best full-5-fold LB scores measured this
campaign, well beyond noise. The fold0/4-only evidence that capacity doesn't help was apparently an
artifact of testing on too few folds, not a real property of the problem. Revised iteration
priorities: (a) pretrained-backbone variants and ensembles thereof, now the leading axis; (b)
history/temporal-architecture reformulation, still open and untested at full scale; (c)
regularization/diverse-ensembling, still relevant but no longer the only live track.

## Verification discipline (every candidate before an LB slot)
1. fold0/4 gate vs 0.28159 / 0.58503 (both improve, or one improves + other within ~0.004 noise).
2. If gate passes → 5-fold OOF; architecture-type OOF is trustworthy (doc/oof_lb_transfer...).
3. `l_eda/exp005/submission_gate.py` GO + `scripts/verify_causal_replay.py` clean before submit.
4. super-site twin grouping for any honest-OOF (doc/twin_fold_and_trap_audit).

## Experiment ledger (update as each completes)

| exp | axis | hypothesis | config | fold0 | fold4 | verdict |
| --- | --- | --- | --- | ---: | ---: | --- |
| exp063 | history | 2h predecessor helps | cr=4, 207ch, stack | 0.28474 | 0.59410 | ✗ both worse (too much/crude) |
| exp064_effv2s | capacity | pretrained beats scratch | effv2s | 0.28208 | 0.58531 | gate said ~tie; **actual 5-fold LB 0.67400, -0.00877 vs old champion — gate was wrong, this wins** |
| exp064_effb3 | capacity | " | effb3 | 0.28833 | 0.58135 | gate said mixed; **actual 5-fold LB 0.67201, -0.01076 vs old champion — NEW CHAMPION, gate was wrong** |
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
| exp056 × effv2s | ensemble | different feature extractor decorrelates errors where seeds could not | equal blend | — | — | built as `diverse_champion_x_effv2s_submission.zip` 07-28, not yet scored — re-prioritize now that effv2s solo is 2nd-best, and consider re-blending against effb3 (the new champion) instead |

## Parallel diagnostic track
- D1: OOF residual anatomy by location/satellite/amount-regime on exp056 (where does it fail?) →
  targets the next architecture iteration. (uses existing analyze_oof outputs; super-site grouped.)
- D2: is the honest ceiling ~0.68 (information-limited) or is 0.63-0.65 reachable? Track vs
  MahmoudElshahed's 0.6798 defensible + his 0.63-0.65 prediction.

## Batch D (2026-07-28) — complete gate-passers to submittable + regularization sweep
Launched after Batch A/B sat idle ~20h unreviewed; results pulled from logs and folded in above.
| exp | axis | hypothesis | jobs | status |
| --- | --- | --- | --- | --- |
| exp064_swin_lr2e4 | complete-to-submit | fold0/4 gate-passed (best gate result since exp056); folds1-3 done | 3952674-76 | **folds1-3 complete as of 2026-07-29 — all 5 folds now trained, not yet built into a submission zip or scored.** Given the 07-30 finding that fold0/4 gate underestimates pretrained/capacity changes, and this one already cleanly PASSED the gate, this is now the top-priority next submission: build zip + score on LB |
| exp064_convnext_lr2e4 | complete-to-submit | NaN-fixed tie; still useful as a diverse-blend member | 3952677-79 | queued |
| exp056_wd5x (weight_decay 5e-4) | regularization | targets diagnosed generalization/distribution-shift bottleneck directly | 3952681-82 (fold0/4) | queued |
| exp056_wd10x (weight_decay 1e-3) | regularization | " (larger step) | 3952683-84 (fold0/4) | queued |

## Submission-ready deliverables (2026-07-30 update)
Already-trained (5-fold) zips built 07-27/28, plus one new diverse blend:
1. ~~`exp064_effv2s_submission.zip`~~ — **SCORED 2026-07-30 07:17:24: 0.67400, -0.00877 vs old champion. 2nd-best result of the campaign.**
2. ~~`exp064_effb3_submission.zip`~~ — **SCORED 2026-07-30 07:16:38: 0.67201, -0.01076 vs old champion. NEW CHAMPION — gate had called this "mixed", actual LB is the best result of the campaign by a wide margin.**
3. ~~`exp056_seed_ens_4best_submission.zip`~~ — **SCORED 2026-07-27 10:14:15: 0.68464, +0.00187 worse than champion. Confirms 4-seed scale-up does not help (matches OOF exhaustion note above). Do not resubmit.**
4. ~~`exp056_seed_ens_6_submission.zip`~~ — **SCORED 2026-07-27 10:14:30: 0.68489, +0.00212 worse than champion. Confirms 6-seed scale-up does not help; seed ensemble plateaus at 2 seeds (42+456). Do not resubmit.**
5. `diverse_champion_x_effv2s_submission.zip` — **NEW** equal blend of champion 2-seed ensemble x effv2s (architecture-diverse, Batch C's actual idea), built 2026-07-28
Remaining unscored (#5, `diverse_champion_x_effv2s_submission.zip`) is not yet run through
`l_eda/exp005/submission_gate.py` (needs OOF, not just fold0/4/eval-only) — do that before
spending an actual LB submission slot on it. It also predates the 2026-07-30 champion change (it
blends the OLD seed-ensemble champion x effv2s, not the new effb3 champion) — see Batch E below for
its replacement.

## Next generations (queued ideas, launch as batches free up)
- Temporal architecture for history (shared per-frame encoder + temporal attention/ConvLSTM) —
  proper use of history vs crude channel-stack.
- Regularization sweep on exp056 (dropout/weight-decay/augment strength) — directly targets the
  generalization bottleneck the diagnosis points to.
- exp056 factorization internals (decoder capacity, loss weights, amount-head design).
- Diverse blend generations as new architecture-distinct members clear the gate.

## Batch E (2026-07-30) — two parallel tracks, launched together

Directive from this session: run an ensemble-refinement track and a new-accuracy-exploration
track concurrently, now that exp064_effb3 is the new champion and the fold0/4 gate is known to
misjudge the pretrained-backbone axis. Both tracks are dependency-chained so each stage only runs
after its prerequisite actually succeeds (`--dependency=afterok:...`), not just after it was
launched.

### Track A — ensemble accuracy (architecture-diverse champion ensemble)

New members added to `g_eda/exp011/sources.json` / `green_allowlist.json` (registry_guard-verified
green in `doc/submission_registry.md`): `exp056`, `exp064_effb3`, `exp064_effv2s`,
`exp064_swin_lr2e4`. `g_eda/exp011/nested_blend.py`'s `fit_weight()` was extended to support N=4
sources via greedy forward blend-in (previously only implemented up to N=3; unit-verified against
synthetic data before use). New `g_experiments/exp065` builds the actual weighted submission zip
from whichever weights `nested_blend.py` recommends (its `in_sample_weights`, gated by its
`nested_score` vs `in_sample_score` overfitting check) — see `g_experiments/exp065/README.md` for
the full pipeline and why this avoids the `exp055` in-sample-only inversion.

Job chain (`afterok` dependencies enforced, not just documented):
1. OOF cache (GPU, `g_eda/exp011/singularity_cache.sh <name>`): exp056=3955562 (done, 126s),
   exp064_effb3=3955563 (done, 126s), exp064_effv2s=3955564, exp064_swin_lr2e4=3955565.
2. exp064_swin_lr2e4 standalone submission build (GPU, `singularity_run.sh config_swin_lr2e4.yaml
   submit` — this experiment had 5-fold checkpoints but had never been through inference.py, so it
   also gets its own solo LB score out of this): job 3955566.
3. `g_eda/exp011/singularity_nested_gate.sh` (CPU-only-workload but this partition's scheduler
   hard-requires `--gpus-per-node>=1` regardless — discovered via a failed `--test-only` submission
   this session; `singularity_analyze.sh` had the identical latent bug and, per the total absence
   of any `slurm-g-eda-exp011-analyze-*.out` log, had apparently never actually been run before;
   fixed both scripts): depends on all of (1), job 3955571. Runs `nested_blend.py --sources exp056
   exp064_effb3 exp064_effv2s exp064_swin_lr2e4` then immediately `l_eda/exp005/submission_gate.py
   --baseline exp064_effb3 --candidate champion_ensemble_nested_blend`.
4. `g_experiments/exp065/singularity_build.sh`: depends on (2)+(3), job 3955572. Builds
   `outputs/submissions/exp065_champion_ensemble[_causal].zip`. **Building the zip is not the same
   as submitting it** — read the gate verdict from (3)
   (`outputs/l_eda/exp005/pairs/champion_ensemble_nested_blend_vs_exp064_effb3/SUBMISSION_GATE.md`)
   and this build's own stdout warning before spending an LB slot on it.

### Track B — new accuracy exploration (backbone scale-up within/beyond the winning families)

Two new `g_experiments/exp064` configs, both registry-eligible extensions of the now-champion
`pretrained_factorized` architecture (no new model code needed — `PretrainedFactorizedUNet`
already takes any timm `encoder_name`):
- `config_effb4.yaml` (`exp064_effb4`): `efficientnet_b4`, one size up from the champion `effb3`,
  same lr=1e-3 regime (EfficientNet family showed no NaN risk at that lr, unlike convnext/swin) —
  tests whether within-family capacity scaling continues to help now that fold0/4 is known
  unreliable for this axis.
- `config_convnext_small_lr2e4.yaml` (`exp064_convnext_small_lr2e4`): `convnext_small`, one size up
  from `convnext_tiny` (which only tied the baseline on fold0/4 at lr=2e-4 — an ambiguous result
  given what we now know about the gate), same lr=2e-4/reduce_on_plateau regime that already fixed
  convnext_tiny's lr=1e-3 NaN divergence.

Both configs were smoke-tested (`_smoke_new_backbones.sh`, job 3955574, **PASSED** — finite
loss/gradients, correct channel math, both loss-ablation toggles clean, in ~90s) before any
fold-training GPU time was committed, and the 5-fold training jobs were submitted with
`--dependency=afterok:3955574` so they could never have started against a config that failed the
smoke test.

Given the round8 plan's own 2026-07-30 finding that a fold0/4 "tie"/"mixed" verdict is not
trustworthy evidence of no value on this axis, **both configs go straight to all 5 folds in
parallel** (not fold0/4-first) — there is no dependency benefit to staggering them since all 5
folds run as independent parallel jobs regardless, and gating on an unreliable partial signal was
exactly this week's mistake.

Job chain (original): smoke=3955574 →
- `exp064_effb4`: folds 3955575/3955576/3955577/3955578/3955579 → submit 3955580
- `exp064_convnext_small_lr2e4`: folds 3955581/3955582/3955583/3955584/3955585 → submit 3955586

**2026-07-30, ~1h into training — cancelled and relaunched with early stopping.** Mid-flight,
inspecting `exp064_convnext_lr2e4`'s and `exp064_effb3`'s own already-completed fold1 histories
(prompted by a direct question about whether the "high plateau" reported earlier was actually
overfitting) showed a clean, textbook signature: `valid_tile_rmse` peaks at epoch 2-3 and never
improves again through epoch 100, while `train_tile_rmse` keeps monotonically decreasing the whole
time (e.g. convnext_lr2e4 fold1: train 0.592→0.435, valid stuck at 0.72-0.82 after epoch 2). This
is real overfitting, not just noise — but it does NOT corrupt any already-scored result, because
`train.py` saves `best_model_fold{N}.pt` synchronously on every validation improvement (confirmed
both in the code, `torch.save` inside the `if current_metric < best_metric` block, and on disk —
the killed jobs' checkpoints were already present and repeatedly updated). The only cost was
~97 epochs/fold of pure wasted GPU-hours with `epochs: 100` and no `early_stopping_patience` set
(every prior `exp064` arm has this same gap — not unique to these two new configs).

Cancelled all 12 jobs from the original chain (`3955575-3955586`, none had reached a fold's own
best epoch yet given the ~1h runtime vs. historical best_epoch=2-3) and added
`early_stopping_patience: 20` / `early_stopping_min_delta: 0.0` to both
`config_effb4.yaml`/`config_convnext_small_lr2e4.yaml` (patience=20 gives >2 full
ReduceLROnPlateau cycles, patience=4 each, to confirm no genuine post-LR-drop recovery — comfortably
past the epoch-2-3 plateau both historical curves actually showed). `train.py`'s checkpoint/log
files are all opened in write/truncate or `best_metric=inf`-reset mode at the start of a fresh run,
so no manual cleanup was needed before relaunching. No re-smoke-test needed (the field is a
purely-additive `config.get(..., 0)` default, no behavior change to anything the smoke test covers).

Relaunched job chain (current, no smoke-test dependency needed a second time):
- `exp064_effb4`: folds 3955671/3955672/3955673/3955674/3955675 → submit 3955676
- `exp064_convnext_small_lr2e4`: folds 3955677/3955678/3955679/3955680/3955681 → submit 3955682

Both submit jobs end by building a standalone submission zip via the existing `run.sh submit` path
(analyze_oof.py → inference.py → make_submission.py) — same as every other `exp064` arm. Score
them on LB once they land; do not assume a good or bad fold-training outcome without checking, per
this session's own lesson about the gate. **Follow-up worth doing later:** the same
`early_stopping_patience` gap exists in every other `exp064`/`exp056` config already in this repo
(e.g. `exp064_swin_lr2e4`'s still-running-to-100-epochs folds) — not fixed retroactively here since
those jobs are already deep into or past their own useful epochs, but worth adding for any future
new arm.
