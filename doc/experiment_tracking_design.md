# Experiment Tracking & Trend-Graphing Design

Last updated: 2026-07-09

Design plan for `I-001` in `doc/task_tickets.md`. Goal: as we run many more cloud (`g_experiments/`)
experiments, keep accumulating their results in a form that is durable, small enough to live in git,
consistent across architectures, and trivially easy to turn into trend graphs — without depending on
HPC group storage or this NAS mount staying alive forever.

## Problem

- `outputs/` (all of it: `analysis_summary.json`, per-sample OOF CSVs, calibration JSON, l_eda
  figures) is fully git-ignored (`.gitignore`: `outputs/`). Nothing under it is version-controlled.
- The only durable, git-tracked score record today is `doc/public_scores.md` (hand-maintained
  markdown table of public leaderboard scores) plus each experiment's `config.yaml`/`README.md`.
  CV/OOF metrics, calibration parameters, and threshold sweeps are not recorded anywhere durable.
- `analysis_summary.json` paths (e.g.
  `/group/project143/yamamoto/solafune_precipitation_nowcasting/outputs/analysis/exp009/...`) show
  results are produced on HPC group storage, then copied to this NAS mount by hand. That copy step
  is undocumented and easy to skip or lose track of.
- The JSON schema itself drifted over the project's life:
  - exp003–006: `training.best_rmse_mean` / `best_rmse_std`, `oof_global.rmse`, no `tile_rmse`.
  - exp008–011: renamed to `training.best_metric_mean` / `best_metric_std`, added
    `selection_metric`, `oof_official_metric`, `oof_global.tile_rmse`.
  - exp007 (ensemble): a completely different shape — a list of source-model summaries, no
    `oof_global` for the ensemble's own blended prediction at all.
- Net effect: today, comparing experiments means opening several JSON files by hand and remembering
  which schema version each one uses. That does not scale past a handful of experiments, and we are
  about to run many more (`exp012`+ in `doc/task_tickets.md` Round 2).

## Goals

1. Every cloud run — single model, two-head, or ensemble — appends exactly **one durable row** to a
   canonical, git-tracked registry, regardless of which `analysis_summary.json` shape it produced.
2. The registry stays small (KB-scale CSV, not the multi-MB per-pixel OOF CSVs) so it can be
   committed to git, diffed in review, and read from a laptop with just `git clone` — no NAS/HPC
   access required.
3. Heavy raw artifacts (per-sample OOF CSVs, calibration sweeps, checkpoints, l_eda figures) keep
   living outside git exactly as today; the registry stores a relative pointer to them, not a copy.
4. One command turns the registry into trend graphs + a short report, runnable locally with only
   pandas/matplotlib — no GPU, no `data/`.
5. Schema drift is fixed going forward via an adapter, and existing exp001–exp011 results are
   backfilled once, not re-entered by hand.

## Canonical Registry Schema

New file: `doc/experiment_registry.csv` (git-tracked). One row per `run_id`.

| Column | Meaning |
| --- | --- |
| `run_id` | `{exp_name}_{kind}`, e.g. `exp009_oof`, `exp009_public`, `exp007_ensemble` |
| `exp_name` | `exp001`, `exp002`, ... |
| `date` | ISO date the row was recorded |
| `git_commit` | short SHA of the repo at analysis time |
| `kind` | `train_oof` \| `ensemble` \| `public_submission` |
| `architecture` | `model.architecture` from `config.yaml` |
| `in_channels` | `model.in_channels` |
| `context_rows` | `data.context_rows` if present, else blank |
| `loss_name` | `loss.name` |
| `selection_metric` | `rmse` or `tile_rmse` (what fold selection was based on) |
| `cv_metric_mean`, `cv_metric_std` | fold-level best-metric mean/std, in the `selection_metric` unit |
| `oof_rmse` | global pixel-pooled OOF RMSE |
| `oof_tile_rmse` | per-sample-averaged OOF RMSE (`mean(sqrt(mean(pixel_error^2)))`, exp008+ only; the closer proxy to how the competition likely scores per file) |
| `oof_mae`, `oof_bias` | from `oof_global` |
| `oof_csi`, `oof_precision`, `oof_recall`, `oof_far`, `rain_prob_threshold` | from `best_rain_threshold`, blank for single-head models |
| `public_rmse` | joined from `doc/public_scores.md` once submitted; blank until then |
| `elapsed_seconds` | wall-clock cost of the analysis/training job, for a cost-vs-quality view |
| `analysis_dir` | relative path, e.g. `outputs/analysis/exp009` (informational — may not exist on every machine) |
| `notes` | free text, e.g. ensemble member list + weights |

Rows are upserted keyed by `run_id`, so re-running an experiment's analysis updates its row in place
instead of duplicating it.

## Schema Adapter (fixes the drift)

A single normalization step maps every known `analysis_summary.json` shape to the canonical columns
above:

| Source shape | Maps to |
| --- | --- |
| exp003–006 (`training.best_rmse_mean/std`, `oof_global.rmse`, no `tile_rmse`) | `selection_metric=rmse`, `cv_metric_mean/std` from `best_rmse_mean/std`, `oof_tile_rmse` left blank |
| exp008–011 (`training.best_metric_mean/std`, `selection_metric`, `oof_official_metric`, `oof_global.tile_rmse`) | `cv_metric_mean/std` from `best_metric_mean/std`, `oof_tile_rmse` from `oof_global.tile_rmse` |
| exp007-style ensemble (`sources: [...]`, no `oof_global`) | `kind=ensemble`, `cv_metric_mean/std` blank, `notes` lists each source's `best_rmse_mean` and weight — **until G-019 adds a real blended `oof_global`, ensemble rows stay metric-incomplete by design; this is the concrete reason G-019 is P0** |

This mapping table itself is the spec for a future `scripts/update_registry.py` — writing that
script is implementation, out of scope for this design note, but the schema above is meant to be
directly implementable without further research.

## Update Workflow

1. A cloud run finishes and its `analyze_oof.py` (or equivalent) writes
   `outputs/analysis/{exp}/analysis_summary.json` as it does today — no change to training/analysis
   code required for this step.
2. A short post-step reads that JSON plus the experiment's `config.yaml`, applies the schema adapter
   above, and upserts one row into `doc/experiment_registry.csv`. This can run:
   - as the last step of `singularity_run.sh` if the container has git and the repo is writable
     there, or
   - as a small separate local/head-node step after the existing NAS copy
     (`/group/project143/.../outputs/analysis/{exp}` → this repo's `outputs/analysis/{exp}`), which
     is already an implicit part of the current workflow (evidenced by the group-storage paths
     baked into the JSON files under this NAS mount).
3. When a Solafune public score comes back, record it in `doc/public_scores.md` as today, then
   backfill the matching registry row's `public_rmse` — same event, two files, kept in sync by the
   same step.
4. Commit `doc/experiment_registry.csv` together with the rest of that experiment's tracked files
   (`config.yaml`, `README.md`, `doc/public_scores.md`). It should stay small enough (one row per
   run) that this is a trivial diff, unlike trying to track the raw OOF CSVs.

## Trend Graphing

A local, no-GPU, no-`data/` script reads only `doc/experiment_registry.csv` and produces:

- RMSE / `tile_rmse` trend across experiments in chronological order — the primary "are we
  improving" chart.
- CV(OOF) vs. public RMSE scatter, to watch the CV/LB gap as more experiments accumulate (this
  project already has one gap surprise — exp001's local CV 0.811 vs. public 0.753 — worth having a
  standing chart for).
- CSI / precision / recall trend across two-head experiments, to track rain-detection quality
  separately from amount-regression RMSE (directly useful for judging G-021).
- Elapsed-time (compute cost) alongside RMSE, to see cost/quality tradeoffs across scaled configs
  (A100x2/x4 tickets).

Because this only touches a small CSV, it belongs alongside the other lightweight, dependency-free
analysis tooling (`eda/`, `l_eda/`) rather than requiring an HPC job — anyone with the repo cloned
can regenerate the current trend picture in seconds. Output PNGs are regenerable, so they can stay
under the existing `outputs/` gitignore; only the registry CSV that produced them needs to be
version-controlled.

## Retention Policy

- `outputs/` (raw per-pixel CSVs, checkpoints, figures) stays git-ignored — it is large and
  regenerable from a checkpoint + code, so git is the wrong home for it.
- The NAS mount (this filesystem) is the de facto durable copy target after each HPC run today;
  that manual/scripted sync (HPC group storage → NAS) should be written down as an explicit step
  wherever `update_registry.py` ends up living, since HPC group directories are typically
  quota-managed and not guaranteed long-term storage.
- `doc/experiment_registry.csv` + `doc/public_scores.md` + each experiment's `config.yaml`/`README.md`
  (all git-tracked, all small) become the things that must survive if both the HPC group directory
  and this NAS mount are ever lost. Together they satisfy the reproducibility bar in
  `doc/competition_rules.md` (experiment ID, git diff, preprocessing, CV split, training/inference
  config, post-processing) without needing the heavy artifacts at all.

## Migration

Backfill `doc/experiment_registry.csv` for exp001–exp011 in one pass once the adapter script exists,
reading the `analysis_summary.json` files already present under `outputs/analysis/*` on this NAS
mount — a one-time scripted run, not eleven rows typed by hand.

## Open Questions

- Whether the Singularity container used by `singularity_run.sh` has `git` and write access to the
  repo to run the update step in-job, or whether it must be a separate head-node/local step. Needs
  checking against the actual container
  (`kaggle-gpu-images-python-v163.sif`) before implementation.
- G-019 (ensemble `oof_global`) is a prerequisite for ensemble rows ever having real metrics in the
  registry, not just member-list notes — sequence that ticket before relying on ensemble trend data.

## Relationship to Existing Docs

- `doc/public_scores.md` remains the source of truth for public leaderboard scores; this design adds
  the CV/OOF side and ties the two together via `public_rmse` backfill.
- `doc/task_tickets.md` (`I-001`) tracks implementing this design; the schema and workflow above are
  the spec for that ticket, not yet implemented code.
- `EXPERIMENT_REPORT.md` is the older, hand-maintained summary — once the registry exists, it can be
  regenerated from `doc/experiment_registry.csv` instead of edited by hand, but that migration is not
  part of this ticket.
