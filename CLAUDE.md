# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Competition Context

Solafune competition "宇宙からの降水ナウキャスト" (satellite precipitation nowcasting). Predict calibrated
GPM-IMERG precipitation rasters from up to 3 recent (≤30 min) multispectral satellite observations
(Himawari 8/9, GOES, or Meteosat — 16 bands each). Metric is RMSE. **External datasets and external
pretrained-data are prohibited** — only the distributed data and features derived from it may be used.
See `doc/overview.md`, `doc/dataset.md`, `doc/competition_rules.md` for details, and
`EXPERIMENT_REPORT.md` for current results and findings.

## Setup

```bash
uv sync
```

Data archives are not stored in git. Place them in `data/raw/` and unzip:

```bash
cd l_experiments/exp000
bash download.sh
```

Expected layout after extraction (see `doc/dataset.md` for full spec):

```text
data/
├── train_dataset/{train_dataset.csv, goes/, gpm_imerg/, himawari/, meteosat/}
├── evaluation_dataset/{evaluation_target.csv, goes/, himawari/, meteosat/, test_files/}
└── sample_submission/{evaluation_target.csv, test_files/}
```

## Commands

Run experiment scripts from inside the experiment's own directory (they use relative paths and local
imports like `from dataset import ...`, `from model import ...`):

```bash
cd l_experiments/exp001
../../.venv/bin/python train.py
../../.venv/bin/python inference.py
../../.venv/bin/python make_submission.py
```

`eda/` scripts run from repo root and depend only on stdlib/NumPy/Matplotlib (no rasterio/pandas
requirement, to keep them runnable in constrained sandboxes):

```bash
python3 eda/01_data_overview.py
python3 eda/02_deep_dive.py
python3 eda/03_satellite_anomalies.py
```

Heavier, modeling-relevant EDA (OpenCV-based morphology/parallax/motion/texture analysis) lives in
`l_eda/expNNN` (local) and `g_eda/expNNN` (HPC), mirroring the `l_experiments`/`g_experiments` split
below:

```bash
cd l_eda/exp001 && bash run.sh                 # or: uv run python l_eda/exp001/run_image_eda.py --config config.yaml
cd g_eda/exp001 && sbatch singularity_run.sh    # HPC
```

There is no lint/test suite configured; validate changes by running the relevant experiment or EDA
script directly.

## Architecture

### Two-tier experiment layout (`l_experiments/` vs `g_experiments/`)

- `l_experiments/expNNN/` — local, CPU-friendly development and correctness checks (small samples,
  low resolution).
- `g_experiments/expNNN/` — GPU/HPC runs of the same experiment number, submitted via
  `singularity_run.sh` (Slurm `sbatch`). Once an `l_experiments` idea is validated, it gets mirrored
  here under the **same experiment number**.
- Each experiment is self-contained: `config.yaml` holds every setting needed to reproduce it
  (paths, model, split, training hyperparameters). Don't hardcode values that belong in config.
- Standard file set per experiment (see `g_experiments/README.md`): `config.yaml`, `dataset.py`,
  `model.py`, `losses.py`, `train.py`, `inference.py`, `make_submission.py`, plus
  `singularity_run.sh` + `python_run.sh` (HPC entrypoints) and `README.md` describing the method.
  Not every experiment needs every file — mirror whichever ones the idea actually touches.
- `g_experiments/expNNN/singularity_run.sh` is an `sbatch` script: it resolves the container at
  `$CONTAINER_FOLDER/$CONTAINER_NAME` (env-overridable, default
  `/group/project143/common/containers/kaggle-gpu-images-python-v163.sif`), binds the project root
  into the container, and runs `python_run.sh <stage>` inside `singularity exec --nv`. Submit with
  `sbatch singularity_run.sh [train|inference|...]`.
- Trained weights go to `l_model/` / `g_model/` (git-ignored, named
  `{exp_name}_{model_name}_fold{fold}_best.pth` per `g_model/README.md`), never into the submission
  zip.
- `experiments/` is legacy; do not add new work there.
- Work is tracked as `L-XXX` (runs under `l_experiments/`, fast local iteration) / `G-XXX` (runs
  under `g_experiments/`, full-scale or scaled-hardware) tickets in `doc/task_tickets.md`. GPU
  scaling work (A100x2/x4) is always G-track. `g_experiments/EXPERIMENT_PLAN.md` and
  `doc/task_tickets.md` track per-experiment status more actively than `EXPERIMENT_REPORT.md` —
  check their "Last updated" dates and prefer whichever is newer when they disagree.

### Data access pattern

- CSV rows are the unit of work. Key columns: `unique_id`, `name_location`, `satellite_target`
  (`goes`/`himawari`/`meteosat`), `datetime`, `last_30_minutes_observation_filename` (a
  **stringified Python list** — parse with `ast.literal_eval`, not JSON), `gpm_imerg_filename`.
- Observation files live under `<split_dir>/<satellite_target>/<filename>`; targets live under
  `<split_dir>/gpm_imerg/<filename>` (train) or must be written to `test_files/` (submission).
- Rows can have fewer than 3 observations (even zero) — datasets must tolerate missing observation
  slots (see `PrecipDataset._input_tensor` in `l_experiments/exp001/dataset.py` for the padding +
  mask-channel pattern: missing slots become zero maps plus a zero mask channel).
- GOES files have shape anomalies: channel counts other than 16 (12–15), and some `282x282x4` files
  that must be resized. Any new loader must keep this padding/resizing logic — see
  `doc/dataset.md` "Checks For Modeling" and `EXPERIMENT_REPORT.md` findings.
- `l_experiments/exp001/tiff_utils.py` is a **dependency-free TIFF reader/writer** (parses IFD tags,
  handles uncompressed + zlib/deflate strips) built specifically so EDA/experiments don't require
  rasterio in constrained environments. `write_float32_like_template` writes predictions by
  patching raster bytes into a copy of a template TIFF, preserving all GeoTIFF metadata — this is
  the required approach for writing submission files.

### CV strategy

Train/evaluation locations do **not** overlap (confirmed by EDA) — never use random row-level CV.
Split by `name_location` (see `make_location_split` in `dataset.py`, `cv.group_column` in
`configs/default.yaml`). Prefer GroupKFold or explicit location holdout.

### Submission format

```text
submission.zip
├── evaluation_target.csv
└── test_files/
    └── {location}_GPM_IMERG_{datetime}.tif
```

Predictions must be non-negative (clip at inference) and preserve the sample/evaluation template
GeoTIFF metadata exactly (dtype, shape, compression must match — `write_float32_like_template`
enforces this and raises if shapes/dtypes mismatch).

### Config precedence

`configs/default.yaml` documents the overall intended architecture (U-Net w/ efficientnet encoder,
48ch input, GroupKFold CV) — treat it as project-level defaults/reference, not as something scripts
load directly. Each experiment's own `config.yaml` is what actually drives that experiment's code and
may diverge (e.g. exp001 uses a from-scratch Compact UNet with 54 input channels, not the
segmentation-models-pytorch Unet from the default config).

### extract_scores.py

Standalone utility (run via `run_extract_scores.sh` on the HPC cluster) that scans `g_model*/` and
`l_model/` for `*.pth` and `*.pt` checkpoints, reads `best_score` (falling back to `best_rmse`), and
prints a per-experiment/fold summary. Update this if training code changes what key the best-metric
value is stored under.
