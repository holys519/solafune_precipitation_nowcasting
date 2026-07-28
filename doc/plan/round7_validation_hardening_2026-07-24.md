# Round 7: validation hardening (2026-07-24)

## Why this round exists

Three submissions in the 2026-07-24 session all showed an OOF improvement that did not transfer to
the public LB:

| Candidate | OOF vs champion | Public LB vs champion | 
| --- | ---: | ---: |
| `exp050_sigmafixed` | −0.00185 (best OOF of the session) | +0.00038 (noise) |
| `exp047_sigmafixed` | −0.00081 | **+0.01350 (worst regression of the project)** |
| `exp055` (`exp038_sigmafixed` × `exp040_metric` blend) | −0.00870 (in-sample fit) | +0.00057/+0.00062 (worse than solo) |

This is not three unrelated bad calls — it is the same structural weakness surfacing three times:

1. **Only 20 train locations back the 5-fold OOF**, and GroupKFold's fold sizes are wildly uneven
   (fold0 has just 2 locations). `l_eda/exp004`'s fold anatomy already showed most of the fold-to-fold
   RMSE spread is explained by which locations/regimes landed where, not by model quality. A single
   OOF point estimate at this sample size has much more sampling noise than its precision (3-4
   decimal places) suggests.
2. **Blend weights were fit in-sample** (`g_eda/exp011`'s `search_two_way`, before this round):
   the same OOF tiles were used to choose the weight and to report the resulting gain. A weight that
   is free to move can always find *some* combination that looks good on the exact sample it was
   fit on; whether that combination reflects real complementary error structure (generalizes) or
   just fold-specific patterns (doesn't) was never tested.
3. **Some candidate features may be location-identity shortcuts, not physics.** `exp047_sigmafixed`'s
   post-mortem hypothesis: `hemisphere` crossed with the satellite one-hot lets the model memorize
   per-(satellite, hemisphere) train-region climate baselines. Since eval locations are guaranteed
   disjoint from train, this kind of "improvement" is definitionally non-transferable, and yet
   nothing in the existing OOF pipeline would have distinguished it from a real signal before this
   round — it needed a specific after-the-fact hypothesis to explain a large LB regression.

## What Round 7 built (`I-002`)

Three genuinely new checks, all pure post-hoc analysis of the `oof_sample_metrics.csv` /
`*_oof_pred.npz` caches that already exist — no retraining, no new GPU compute.

### 1. Location-cluster bootstrap CI (`l_eda/exp005/bootstrap_ci.py`)

Treats each of the 20 locations as one resampling unit (a block/cluster bootstrap — tiles inside
one location are not independent draws, they share a fold, a satellite, a climate regime), resamples
locations with replacement, and reports the empirical distribution of the candidate-vs-baseline
delta. If the 80% interval straddles zero, the point estimate is not distinguishable from
fold-composition noise, no matter how good it looks as a single number.

### 2. Gain-concentration / geography-shortcut audit (`l_eda/exp005/leakage_audit.py`)

Breaks the candidate's improvement down per location (and per satellite) and checks how concentrated
it is. A real, generalizable improvement should show up diffusely across most locations; an
improvement that is actually 70-90% attributable to 1-2 specific locations is the exact signature of
a location-identity shortcut, independent of whether we can name the mechanism.

### 3. Outer-cross-fit (nested) blend optimizer (`g_eda/exp011/nested_blend.py`)

Reuses the existing 5-fold split as the outer CV unit: for each outer fold, fits the blend weight
using only the *other* four folds' OOF tiles, then scores that weight on the held-out fold — the
weight-fitting for a fold's tiles never sees those tiles. The gap between this nested score and the
naive in-sample score (what `optimize_blend.py --analyze` reports) is a direct, submission-free
measurement of how much the in-sample number was overfitting fold structure.

### 4. Submission gate (`l_eda/exp005/submission_gate.py`)

Combines (1) and (2) into one verdict:

- **NO-GO** — `|delta| < 0.004` (the noise floor `l_eda/exp003` measured from 12-16 submitted pairs),
  or the 80% bootstrap CI includes zero.
- **HOLD** — clears both of the above, but the gain is concentrated in ≤2 locations at >60% share;
  worth a manual look (does the concentration make physical sense?) before spending a submission.
- **GO** — clears the noise floor, the CI excludes zero, and the gain is diffuse.

## Retroactive validation: would this have caught the three misses?

| Candidate | Gate verdict | Why | Matches what actually happened? |
| --- | --- | --- | --- |
| `exp050_sigmafixed` | **NO-GO** | Δ below noise floor; 80% CI `[-0.00745, +0.00345]` includes zero; top-2 locations (`guangdong`, `jamaica`) are 69% of the positive gain | Yes — realized LB delta (+0.00038) was noise |
| `exp047_sigmafixed` | **NO-GO** | Δ below noise floor; 80% CI `[-0.00651, +0.00517]` includes zero; top-2 locations (`guangdong`, `ecuador`) are 74% of the positive gain — strongly concentrated | Yes — realized LB delta was the project's worst regression (+0.01350), consistent with a shortcut that doesn't generalize |
| `exp055` blend | **GO (would NOT have caught it)** — see below | Nested CV was built specifically to close this gap; it did not | **No** |

Both single-model cases would have been screened out *before* spending a submission, for the
project's own stated reasons after the fact — the gate makes that reasoning mechanical and
pre-emptive rather than a retrospective diagnosis. Notably, `guangdong` is the #1 location driver of
the "improvement" in **both** unrelated candidates (`exp050_sigmafixed` and `exp047_sigmafixed`) —
this is itself a flag worth following up: either `guangdong` has some data-quality idiosyncrasy that
several unrelated small changes happen to exploit, or fold1's small size makes it unusually easy to
move. Not investigated further in this round; recorded here so it isn't lost.

## The exp055 blend is the important negative result: the new toolkit did NOT catch it

Running `g_eda/exp011/nested_blend.py` on the actual `exp055` pair
(`exp038_sigmafixed` x `exp040_metric`) gives a very different picture than the original hypothesis
in `doc/public_scores.md` assumed:

- naive in-sample fit (what `optimize_blend.py --analyze` originally reported): **0.59982**
  (−0.00790 vs best solo, `exp040_metric` at 0.60772)
- **nested (outer-cross-fit) score: 0.60007** (−0.00766 vs best solo) — almost identical to the
  in-sample number. Overfitting gap is only **−0.00024**, and the per-outer-fold weight is stable
  (exp038_sigmafixed's share: 0.45–0.53 across the 5 folds, std 0.029).
- Feeding this nested result through `submission_gate.py` (as `exp038_sigmafixed_nested_blend`)
  returns **GO**: Δ−0.00821 clears the noise floor, the 80% bootstrap CI is
  `[-0.01123, -0.00552]` (excludes zero, P(candidate better)=1.000 over 20000 resamples), and the
  gain is diffuse (15/20 locations improve, top-2 location share only 48%, well under the 60%
  concentration flag).

**The real submission was worse than solo champion by +0.00057–0.00062.** Neither the in-sample fit,
the nested cross-fit, nor the bootstrap CI or concentration audit saw this coming — all four
diagnostics agree the blend should have worked, cleanly and by a wide margin.

This is a structurally different failure from the other two. `exp050_sigmafixed` and
`exp047_sigmafixed` failed because their apparent gain was an artifact of *which 20 train locations*
existed and *how they fell into folds* — exactly what location-cluster resampling is built to expose.
The `exp055` blend's problem is different: the complementary error structure between
`exp038_sigmafixed` and `exp040_metric` may be **real and stable across all 20 train locations**, and
still not hold on the 18 eval locations, because those are a disjoint population with a different
regime mix. Resampling the train locations — however cleverly — samples from the train distribution
only; it structurally cannot detect a train-vs-eval distribution shift, because the eval locations
never enter the calculation at all. Bootstrap CI and nested CV both answer "is this robust *within
what we can measure*" — not "will this hold under the distribution shift we know exists but cannot
observe."

The one piece of evidence that *was* available beforehand and points the right direction:
`exp040_metric`'s own solo OOF-to-LB gap is larger than `exp038_sigmafixed`'s (OOF 0.60772 -> LB
0.69552, a ~0.088 gap, vs. exp038_sigmafixed's OOF 0.60828 -> LB 0.68664, a ~0.078 gap) — i.e.
`exp040_metric` was already known to generalize worse per unit of OOF improvement than
`exp038_sigmafixed`, per `l_eda/exp003`'s established OOF->LB regression. `doc/public_scores.md`'s
original hypothesis #2 said almost exactly this before this round confirmed hypothesis #1
(in-sample overfitting) was NOT actually the operative mechanism here.

**New rule this finding motivates (added to I-002's scope for future blend candidates):** before
trusting any blend's in-train diagnostics (in-sample OOF, nested CV, or bootstrap CI), check each
component's own historical OOF->LB transfer efficiency via `l_eda/exp003`'s regression
(`LB ≈ 1.268×OOF − 0.080`). A component whose solo track record already shows a worse-than-typical
OOF->LB gap should be treated as a generalization risk for the blend as a whole, regardless of how
good the blend's own in-train numbers look — no in-train resampling method can substitute for actual
LB track record when the concern is train-vs-eval distribution shift rather than train-side sampling
noise. This is a genuine, documented limitation of the toolkit, not a solved problem — flagged here
rather than glossed over.

## Policy going forward (through the ~2026-08-03 deadline)

1. **Every candidate with OOF gain < ~0.01 vs the current champion must pass `submission_gate.py`
   before it gets a submission slot.** This screens the fold-composition-noise and
   geography-shortcut failure modes (caught `exp050_sigmafixed` and `exp047_sigmafixed`
   retroactively). Submissions are the scarce resource with ~10 days left; two were already spent
   this session on candidates the gate would have rejected outright.
2. **Every future blend must be scored by `nested_blend.py` in addition to
   `optimize_blend.py --analyze`**, but understand what it does and does not cover: it removes
   in-sample weight-fitting bias (the mechanism the team originally suspected for `exp055`), but
   **it is not sufficient on its own** — `exp055` itself passed nested CV and the bootstrap gate
   cleanly and still lost on LB. It answers "is this weight choice robust within the train
   locations we have," not "will this hold on eval's disjoint locations."
3. **Before adding a component to any blend, check that component's own solo OOF->LB transfer
   efficiency** against `l_eda/exp003`'s regression (`LB ≈ 1.268×OOF − 0.080`, residual std
   ~0.004). A component with a worse-than-typical historical OOF->LB gap (as `exp040_metric` had
   relative to `exp038_sigmafixed`) is a distribution-shift risk for the blend even when the
   blend's own in-train diagnostics look clean. This is the practical, if partial, mitigation for
   the gap `nested_blend.py` cannot close.
4. Locations that repeatedly show up as top gain drivers across unrelated candidates (`guangdong` so
   far) should be treated with extra suspicion — a real per-location EDA pass is a good candidate for
   a future round if time allows, but is not blocking this round's toolkit.
5. This does not replace 5-fold OOF as the primary selection metric (`l_eda/exp003` already showed it
   is the best available predictor of LB, Spearman 0.95) — it adds a confidence check *around* that
   metric before it gets to consume a submission. It closes two of the three failure modes seen this
   session (fold-composition noise, geography shortcuts); the third (train-vs-eval distribution
   shift on an otherwise-legitimate blend) remains open and should not be considered solved.
