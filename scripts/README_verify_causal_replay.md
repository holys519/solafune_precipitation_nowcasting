# Causal-replay compliance verification

Mechanical (not code-review-only) verification that an experiment's inference-time input
construction cannot depend on data timestamped after each prediction's target time T, per the
organizers' 2026-07-20 final ruling (`doc/submission_registry.md`):

> 検証: 勝者のコード検査で「各予測時刻Tより前のデータのみに切り詰めて再実行し、提出結果と
> 一致するか」を実際に検証する。再現できない/上記に違反する解は失格。

i.e. winners get their code inspected: the organizers truncate all data to timestamps <= each
prediction's T, re-run the pipeline, and check the result matches the submission for that row. If
it doesn't reproduce -- or the pipeline turns out to have used data timestamped > T anywhere --
the winner is disqualified. As of 2026-07-22 this project had never actually run that check on its
own pipeline; it was a documented risk in the registry table, not a verified property. These two
scripts close that gap.

## The two tools

| Script | What it does | Cost | Definitiveness |
| --- | --- | --- | --- |
| `audit_submission_config.py` | Greps `config*.yaml` + the experiment dir's `.py` files for red-flag settings (`context_rows > 1`, overlap/patch imports or artifacts, non-causal `temporal_smoothing`), and cross-checks the result against `doc/submission_registry.md`'s classification for that experiment | milliseconds, no GPU, no data needed | Screening only -- can't catch a leak that isn't visible in config/imports |
| `verify_causal_replay.py` | Imports the experiment's *real* `dataset.py` (`PrecipDataset`, `_input_tensor`, `_context_rows`, `read_rows` -- nothing reimplemented) and, for a sample of rows, builds the input tensor twice: once normally, once with the CSV rows list hard-truncated to `datetime <= T` and the file-read choke point patched to raise if ever asked for a path timestamped > T. Asserts `torch.equal` (bit-identical, not `allclose`) between the two. Optionally also compares the deterministic forward-pass output of a fixed `.pt` checkpoint on both inputs. | seconds to a few minutes for ~20-50 rows, CPU only (no CUDA needed unless you insist on GPU checkpoints) | This is the actual organizer audit, run against your own code, now, instead of waiting to find out at judging time |

Run the cheap one first; if it's clean, run the expensive one for the real proof. A config can
pass the cheap grep and still fail the replay if a leak isn't expressible as one of the three
known red-flag patterns -- that's exactly why the replay tool exists.

## Requirements

Both scripts need `PyYAML`. `verify_causal_replay.py` additionally needs `PyTorch` (CPU build is
enough -- none of this repo's `model.py` files hardcode `.cuda()`, so the input-tensor check and
even the optional checkpoint-output comparison run fine without a GPU). If you're on a machine
without the project's usual `.venv` (e.g. a bare login node, not inside the
`kaggle-gpu-images-python-v163.sif` container `singularity_run.sh` normally runs in), a throwaway
CPU install is enough just for running this audit:

```bash
pip install --target /tmp/verify_pylibs torch --index-url https://download.pytorch.org/whl/cpu pyyaml
PYTHONPATH=/tmp/verify_pylibs python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp038
```

Inside the project's normal `.venv` / the Singularity container, just:

```bash
../../.venv/bin/python scripts/verify_causal_replay.py --exp-dir g_experiments/exp038
```

(paths are resolved relative to `--exp-dir`, same convention as every experiment's own
`inference.py`, so this works from the repo root regardless of which experiment you point it at).

## Usage

```bash
# cheap static screen
python3 scripts/audit_submission_config.py --exp-dir g_experiments/exp038
python3 scripts/audit_submission_config.py --exp-dir g_experiments/exp038 --all-configs   # every config_*.yaml too

# real replay audit (default: 30 rows sampled from evaluation_target.csv, stratified across
# (location, satellite) groups)
python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp038

# larger sample, fixed seed for reproducibility, plus a full checkpoint forward-pass check
python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp038 --num-rows 50 --seed 0 \
    --checkpoint ../../g_model/exp038/best_model_fold0.pt

# audit a non-default config variant inside the same experiment dir
python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp038 --config config_sigmafixed.yaml
```

Exit code `0` + a PASS summary table = every sampled row's input tensor (and model output, if
`--checkpoint` was given) is unaffected by whether data timestamped after T is available at all.
Exit code `1` + a detailed per-row failure report = at least one row's construction depends on
future data; the report names the exact context-row offset, the exact file path(s) with
out-of-window timestamps, and the exact flat channel indices (decoded to
`context_row`/`obs_slot`/`ch` where possible) that differ between the normal and truncated
construction.

## Validated against known-good and known-bad experiments (self-test baseline)

Anyone re-running this later can reproduce these two reference results to confirm the tool itself
hasn't regressed:

- **Positive control -- `g_experiments/exp038`** (green strict champion, `context_rows: 1`, no
  overlap patch, no temporal smoothing): both scripts **PASS** cleanly (exit 0). 20/20 sampled
  rows bit-identical between normal and hard-truncated construction; static scan finds no red
  flags and agrees with the registry's `green` classification.
- **Negative control -- `g_experiments/exp018`** (and, checked separately, `exp016`;
  `context_rows: 2`, successor-row input): both scripts **FAIL** (exit 1). 20/20 (resp. 10/10)
  sampled rows fail every one of the three signals at once: `_context_rows` pulls in a CSV row
  dated 30 minutes after T; the normal-path file-access log shows two real reads of `.tif` files
  timestamped 10 and 20 minutes after T; the hard-truncated reconstruction's input tensor differs
  from the normal one at all `frame_channels` of `map[context_row=1, obs_slot=0, ...]` (i.e. the
  entire newest observation slot of the successor row). Example trace (abbreviated):

  ```
  === a695-44d4 (location=valencia, satellite=meteosat, T=2025-12-11T12:00:00) ===
    - context_rows offset=1 pulls row '42d4-5364' dated 2025-12-11T12:30:00 (> target T) -- this is a successor-row / future-data dependency
    - normal-path read '.../meteosat/test_valencia_Meteosat_20251211_1210.tif' timestamped 2025-12-11T12:10:00 (> target T)
    - normal-path read '.../meteosat/test_valencia_Meteosat_20251211_1220.tif' timestamped 2025-12-11T12:20:00 (> target T)
    - input tensor differs between normal and truncated construction at: map[context_row=1,obs_slot=0,ch=0..11] (+39 more)
  ```

  The static scan independently flags `data.context_rows = 2` and agrees with the registry's
  `red` classification for both.
- Overlap-patch detection was validated against the real `g_experiments/exp014` (correctly flags
  `apply_overlap.py` and `overlap_pairs.csv` by filename, agrees with the registry's `red`
  classification) plus a synthetic experiment dir with `import apply_overlap` /
  `from overlap_patch import apply_overlap_patch(...)` (correctly flags both, including the
  substring-only match that a naive `\boverlap\b` word-boundary regex would miss). Non-causal
  temporal-smoothing detection (`next_weight > 0`) and the causal-only permitted case
  (`next_weight == 0`) were validated the same way, via a synthetic config, since no on-disk
  `config.yaml` in this repo currently has `temporal_smoothing.enabled: true` (exp036/037's
  bidirectional smoothing was applied by a standalone blend script outside this config schema, so
  it's out of this tool's reach by construction -- see Caveats).

## Before the ~2026-08-03 deadline

**Re-run `verify_causal_replay.py` (and `audit_submission_config.py`) one final time against
whichever `g_experiments/expNNN` directory and config actually becomes the final submission
candidate, once that decision is made.** As of this writing (2026-07-22) that decision has not
been made yet -- `exp038_sigmafixed` is the current green champion in `doc/public_scores.md`, but
Track G3 blending work (see `doc/submission_registry.md` items 5-6, and the growing green family
`exp038_seed123`/`exp038_seed456`/`exp040_metric`/`exp045`/`exp047`/`exp048`/`exp049`/`exp050`) is
still producing new green candidates in parallel, and the eventual submission may be a blend of
several of them rather than a single experiment directory. Recommended final-check sequence:

1. `python3 scripts/audit_submission_config.py --exp-dir <final_exp_dir> --all-configs`
2. `python3 scripts/verify_causal_replay.py --exp-dir <final_exp_dir> --num-rows 50 --checkpoint <the exact checkpoint(s) used for the submitted predictions>`
3. If the final submission is a **blend** of multiple green experiment directories, run step 2
   separately against *each* source experiment directory that contributes to the blend -- this
   tool audits one `dataset.py`/config pair at a time by design (mirrors how the organizers'
   audit would inspect each model's own code), and a blend is only as clean as its worst
   ingredient (`doc/submission_registry.md` rule 1: "green系の学習・OOF・blend・calibrationに
   amber/red artifactを混入させない").
4. Keep the PASS output (or save it to a file) as the evidence trail that this was actually
   checked before the deadline, not just documented as a risk.

## Caveats / known limitations (read before trusting a PASS blindly)

- **Scope is input construction, not the full submission pipeline.** This audits whether the
  *input tensor* (and, optionally, a fixed checkpoint's deterministic output on that tensor) for
  row T could possibly depend on data timestamped > T. It does not re-run training, and it does
  not diff the actual submitted `.tif` bytes against a fresh end-to-end `inference.py` run --
  that's what running `inference.py` itself is for. If the final pipeline does anything
  cross-row *after* the per-row model forward pass (e.g. `apply_temporal_smoothing` in
  `inference.py`, which operates on the whole `prediction_items` list), that step is **not**
  covered by this tool and must be checked separately -- which is exactly why
  `audit_submission_config.py` has its own dedicated `temporal_smoothing.next_weight` check.
- **Blend / overlap-patch scripts outside the standard `expNNN/{config.yaml,dataset.py,
  inference.py}` layout are out of reach.** exp036/037's bidirectional smoothing and exp014's
  `apply_overlap.py` patching are standalone scripts (`run.py`, `apply_overlap.py`) that don't use
  this repo's standard per-experiment config schema; `audit_submission_config.py`'s
  `temporal_smoothing`/`overlap` checks are written against that standard schema and validated
  against synthetic configs reproducing the same settings, not against those scripts directly (its
  filename-substring check does catch `apply_overlap.py` itself by name, which is why the exp014
  validation above works, but a differently-named future blend script performing the same
  operation would need either a rename-aware update to `OVERLAP_PATCH_NAME_RE`/
  `OVERLAP_PATCH_CALL_RE` or a manual look).
- **Channel-to-source attribution is best-effort.** `verify_causal_replay.py` decodes a flat
  channel index back to `context_row`/`obs_slot`/`ch` by assuming the
  `maps + masks [+ row_features] + satellite_onehot` concatenation order used by every
  `_input_tensor` in this repo's exp009/016/017/018/035/038/040/045 family. The flat index itself
  is always correct (it comes directly from `torch.equal`'s diff); only the human-readable label
  is order-dependent. If a future experiment's `_input_tensor` changes that order, cross-check the
  label against that experiment's own code before trusting it at face value.
- **Global CSV truncation, not per-location.** The hard-truncation reconstruction drops *any* row
  (any location) with `datetime > T`, not just rows for the same location as the row under test.
  This is a conservative superset of "just this row's own history" and matches the organizers'
  literal wording most closely, but it does mean a single truncated-dataset object is rebuilt per
  sampled row (cheap here -- rebuilding the `_row_by_location_time` dict from ~20-29k CSV rows is
  a sub-second operation in Python -- but worth knowing if you scale `--num-rows` way up).
- **"Hiding" files is done by intercepting the read call, never by touching the filesystem.** The
  task framing suggested "delete/hide" the out-of-window observation files; this script
  deliberately does not do that, since the data directories are real, shared, read-only
  competition data and destructive testing on them would be unacceptable even if reversible in
  principle. The guarded reader in the truncated-construction pass raises on any attempt to read a
  path whose embedded filename timestamp is > T, which is equivalent in effect (the code cannot
  get real data from that path) without ever mutating anything under `data/`.
- **Requires PyTorch; none was pre-installed in the environment this tool was built and tested
  in.** A CPU-only `pip install torch --index-url https://download.pytorch.org/whl/cpu` was
  sufficient (no CUDA, no GPU needed) since `model.py` in the audited experiments imports only
  `torch`/`torch.nn`/`torch.nn.functional` (no `segmentation_models_pytorch`, no pretrained-encoder
  download) -- this may not hold for every future experiment's `model.py`, so re-check that import
  list before assuming the `--checkpoint` path will run without a GPU or network access.
