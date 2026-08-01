# exp065: architecture-diverse champion ensemble

2026-07-30: `exp064_effb3` overturned the fold0/4 gate's "mixed" verdict and became the new green
champion on the full 5-fold LB (0.6720097338314985, -0.01076 vs the prior champion
`exp056_seed_ens_42_456`). `exp064_effv2s` is 2nd (0.6740019855472996). See
`doc/public_scores.md` (2026-07-30 追記5) and `doc/plan/round8_campaign_2026-07-27.md`'s
"2026-07-30 update" for the full context, including why the fold0/4 gate underestimated this axis.

The round8 plan's priority (b) follow-up: ensemble the new champion with an architecturally
*different* model, on the same variance-reduction logic that made the 2-seed
`exp056_seed_ens_42_456` beat its solo members -- but along the architecture axis instead of the
seed axis, since `exp056_seed_ens_4best`/`exp056_seed_ens_6` already proved seed-axis scale-up is
exhausted (both scored worse than the 2-seed ensemble on LB, 2026-07-27).

## Members

| name | architecture | role |
| --- | --- | --- |
| `exp056` | from-scratch CompactUNet | most architecturally distinct from the exp064 family; the original champion |
| `exp064_effb3` | timm `efficientnet_b3` | new solo champion |
| `exp064_effv2s` | timm `tf_efficientnetv2_s` | 2nd-best solo, same encoder family as effb3 (lower diversity value, kept for completeness) |
| `exp064_swin_lr2e4` | timm `swin_tiny_patch4_window7_224` (transformer) | most architecturally distinct *within* exp064; best fold0/4 gate result since exp056; 5-fold trained but not yet solo-submitted |

All four are registered in `g_eda/exp011/sources.json` / `green_allowlist.json` and independently
re-verified green (`registry_guard.py`) at both blend-analysis time and submission-build time.

## Pipeline (in order)

1. `g_eda/exp011/singularity_cache.sh <name>` for each of the 4 sources -- GPU, regenerates each
   source's fold-holdout OOF predictions from its checkpoints (inference only, no training).
2. `g_experiments/exp064/singularity_run.sh config_swin_lr2e4.yaml submit` -- GPU, exp064_swin_lr2e4
   has never been through inference.py/make_submission.py; this produces its
   `outputs/submissions/exp064_swin_lr2e4/` eval predictions (needed to blend it at serving time)
   and its own standalone submission zip (worth submitting on its own merits too, given its
   fold0/4 gate result and the now-established unreliability of that gate for this axis).
3. `g_eda/exp011/singularity_nested_gate.sh` -- CPU, after (1): runs `nested_blend.py --sources
   exp056 exp064_effb3 exp064_effv2s exp064_swin_lr2e4`, the OUTER-cross-fit weight search (see
   that file's docstring for why in-sample-only fitting is what caused the exp055 postmortem
   inversion), then immediately runs `l_eda/exp005/submission_gate.py --baseline exp064_effb3
   --candidate champion_ensemble_nested_blend` against the new champion.
4. `singularity_build.sh` (this dir) -- CPU, after (1)+(2): blends the 4 members' eval TIFFs using
   the `in_sample_weights` `nested_blend.py` fit, applies the causal-smoothing hook if
   `g_eda/exp010` has published one, writes `outputs/submissions/exp065_champion_ensemble.zip`.

Building the zip does NOT submit it to Kaggle -- read step 3's gate verdict
(`outputs/l_eda/exp005/pairs/champion_ensemble_nested_blend_vs_exp064_effb3/SUBMISSION_GATE.md`)
and this build's own stdout warning (which fires if the nested score doesn't beat the best solo
member) before spending an actual submission slot on it.

## Why this differs from exp055

`exp055` (superseded) blended `exp038_sigmafixed` x `exp040_metric` using ONLY
`optimize_blend.py`'s in-sample fit and inverted on LB as a result (postmortem in
`doc/public_scores.md`, 2026-07-22). This experiment instead treats `nested_blend.py`'s honest
outer-cross-fit score as the thing to trust before spending a submission, and only uses its
`in_sample_weights` (fit on the same full data, for maximum sample efficiency) as the actual
deployed blend weights once the nested score has been checked.
