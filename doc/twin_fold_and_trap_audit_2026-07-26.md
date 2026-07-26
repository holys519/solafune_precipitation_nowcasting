# Training-side trap audit vs MahmoudElshahed's post (2026-07-26)

Response to the forum post "The Leaderboard is Not What You Need" (MahmoudElshahed), which documents
three training-side traps. Below: our status on each, verified against our own pipeline.

## ① Copy trap (§2) — CLEAN (was never in our final path)
overlap patch / soft-retrieval / ≥T serving / non-causal smoothing are all prohibited. Our champion
`exp056` and the seed ensemble use none of them: `context_rows: 1` (no successor/≥T frames), no
overlap patch (that was `exp014`, red, never in final), no cross-prediction smoothing. Confirmed by
`scripts/verify_causal_replay.py` (exp056: 40/40 sampled rows bit-identical under T-truncation) and
`scripts/audit_submission_config.py` (no red flags). Allowed-list reaffirmed: train imagery for any
purpose, eval-input imagery for preprocessing stats / unsupervised pretraining (not placeholder
targets), per-band norm constants from either split, GeoNames/Nominatim geocoding.

## ② Twin trap (§4–6) — CONFIRMED PRESENT in our CV (OOF-only impact)
GPM-IMERG tiles are integer crops of one 0.1° grid, so two training sites whose crop windows overlap
carry **pixel-identical labels** in the overlap. The post identifies two twin pairs among our 20
training sites: `atlantic_coast↔florida` (~37% overlap) and `bihar↔dhaka` (~13%, newly reported).

**Our seed42 fold assignment (the champion's CV) splits BOTH pairs across folds:**

| Twin pair | fold A | fold B | split? |
| --- | --- | --- | --- |
| atlantic_coast ↔ florida | fold2 | fold4 | **YES ❌** |
| bihar ↔ dhaka | fold3 | fold4 | **YES ❌** |

So for those folds, validation tiles are partly pixel-identical to training tiles → **our OOF has
been optimistically biased** (the post measured ~+0.02 optimism on the held-out twin regions; on the
dry bihar/dhaka pair split-fold roughly halved the apparent error).

**Impact assessment — bounded, and it does NOT threaten the champion:**
- **LB is unaffected** (this is a train-side validation artifact; eval predictions and the 0.68277 /
  0.68396 LB numbers are clean).
- **Compliance is unaffected** (not a causality/copy issue).
- **Relative gating within seed42 is mostly preserved** — every seed42 experiment shares the same
  split, so the twin optimism is ~constant across them; exp056's GO vs exp038_sigmafixed etc. stand.
- **It inflates ABSOLUTE OOF** (our true honest OOF is likely ~0.61–0.62, not 0.60252) → reinforces
  the standing policy: **select finals on LB, not OOF.**
- **It partly explains the large seed OOF variance** (seed42 0.60252 vs seed456 0.61621): different
  seeds shuffle locations differently, so twins land together or split differently → different
  optimism per seed. This is another reason the seed ensemble was correctly selected on **LB**, not
  OOF.

**Fix (for final honest-OOF evaluation and future rounds), one line — NOT applied to the training
path now because seeds 789/1337/2024 are mid-flight and changing `dataset.py` would desync their
folds:**
```python
SUPER_SITES = {"florida": "atlantic_coast", "dhaka": "bihar"}  # keep each twin pair in one fold
group_key = lambda loc: SUPER_SITES.get(loc, loc)   # split on group_key, keep real names on rows
# + assert twins never cross folds
```
Apply this in `l_eda/exp005`'s honest-OOF/gate tooling and in any post-deadline clean retrain.

## ③ Fallback trap (§7) — CLEAN
The post's leak was a missing-frame fallback reaching to the nearest frame in *either* direction
(sometimes future). **Ours is zero-fill + mask=0** for missing slots (`dataset.py` lines ~380/390),
never a nearest-frame lookup, and context_rows:1 forbids any future access. Their training-cache
all-zero-label bug is N/A for us: we load targets per-`__getitem__` from disk, no preallocated cache.

## §9 nulls + transfer rule — independent corroboration of our methodology
Their nine null experiments (blur, sharpen, gamma, warp, physics-composite corrector/input,
autoregressive self-blending, motion/advection, over-strong smoothing) match our own closed
experiments (exp053 autoregressive, exp059 NODATA feature, causal-smoothing marginal) and the
`doc/domain_knowledge_review` optical-flow rejection. Their measured held-out→public transfer
(150%/36%/5%/sign-flip; "treat a held-out win as ~1/3 on the board") is the same phenomenon as our
`doc/oof_lb_transfer_by_category_2026-07-25.md`. Two independent teams, same conclusion:
feature/post-processing tweaks are exhausted; real movement comes from validation you can trust +
genuinely diverse ensembling.

## Frontier calibration
Their code-review-defensible artifact = **0.6798**; they predict the verified winner lands in
**0.63–0.65**. Our current defensible champion (seed ensemble) = **0.68277**, i.e. slightly ahead of
their stated defensible number and near the top of the *known* honest frontier — but if a 0.63–0.65
honest solution exists, there is a real ~0.03–0.05 gap we have not found (vs the measurement-audit
thread's "information-limited" view, under which ~0.68 may be near the honest ceiling). We cannot
resolve which is true; the endgame bets accordingly (ensemble + clean finalization, not more
feature chasing).

## Endgame actions from this post
1. **Finals must be defensible** — our champion is (verified clean on ①/③). Keep it.
2. **Select on LB, never twin-inflated OOF** — already policy; this hardens it.
3. **Super-site grouping** for any final honest-OOF check and next round (fix above).
4. Stop feature/post-processing tweaks (their §9 + our meta-analysis agree). Spend remaining time on
   the seed ensemble and clean finalization (full-data retrain, manifest, per-member causal-replay).
