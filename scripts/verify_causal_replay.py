#!/usr/bin/env python3
"""Mechanically audit an experiment's inference-time input construction for causality leaks.

Why this exists
----------------
The organizers' 2026-07-20 ruling (see `doc/submission_registry.md`) states the winner-audit
procedure verbatim:

    "検証: 勝者のコード検査で「各予測時刻Tより前のデータのみに切り詰めて再実行し、提出結果と
    一致するか」を実際に検証する。再現できない/上記に違反する解は失格。"

    i.e. for each prediction's target timestamp T, the organizers truncate ALL data to
    timestamps <= T and re-run the pipeline; if the result does not reproduce the submitted
    prediction -- or the pipeline turns out to have used any data timestamped > T -- the
    winner is disqualified.

This project has never mechanically run that check on itself; it has only been a documented
risk (`doc/submission_registry.md`'s green/amber/red table, built from code review). This
script performs the actual audit: for a sample of `evaluation_target.csv` rows it builds the
model input tensor TWICE using the *real* project code (imports `dataset.py` directly from the
target experiment directory -- `PrecipDataset`, `_input_tensor`, `_context_rows`,
`_observation_tensors`, `read_rows`, `parse_observation_filenames`; nothing is reimplemented):

  (a) the normal path, exactly as that experiment's own `inference.py` builds it.
  (b) a hard-truncation path: the in-memory CSV rows list is filtered down to
      `datetime <= T` before the dataset is constructed (so any cross-row lookup for a
      timestamp > T -- e.g. a successor row -- returns nothing), AND the file-reading choke
      point (`dataset.read_tiff_array`) is patched to hard-raise if it is ever asked for a
      path whose embedded timestamp is > T. Real project data files are never modified,
      renamed, or deleted -- "hiding" is done by intercepting the read call, not by touching
      the filesystem, which is what makes this safe to run against read-only production data.

Three independent signals are combined per sampled row (any one failing fails the row):
  1. cross-row check: do any of the context rows the dataset's own `_context_rows` returns
     have a datetime > T?  (Directly catches successor-row designs like exp016/exp017/exp018.)
  2. file-access check: during the NORMAL construction, every file the code actually opened is
     logged (non-blocking) and its embedded timestamp compared to T. Any read of a file dated
     after T is reported by exact path.
  3. dual-construction tensor identity: are the normal-path and hard-truncated-path input
     tensors bit-identical (`torch.equal`, not `allclose`)? A mismatch is reported by exact
     flat channel index, decoded back to (context_row, observation_slot, channel) using the
     project's own `channels_per_frame`/`channels_per_row` accounting where available.

Usage
-----
    python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp038
    python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp016 --num-rows 40
    python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp038 \\
        --config config_sigmafixed.yaml --checkpoint ../../g_model/exp038/best_model_fold0.pt

Exit code 0 and a PASS summary table if every sampled row passes all three checks.
Exit code 1 and a detailed failure report (which check failed, which file/row, which channel)
if any row fails.

Caveats (documented, not swept under the rug)
----------------------------------------------
- Requires PyTorch (CPU is enough -- no GPU/CUDA needed for the tensor-construction check;
  only the optional `--checkpoint` model-output comparison exercises the model's forward pass,
  also on CPU, since none of these projects' `model.py` hardcodes `.cuda()`).
- Channel-to-source attribution (signal 3) assumes the `maps + masks [+ row_features] +
  satellite_onehot` concatenation order used by every `dataset.py` `_input_tensor` seen in this
  repo so far (exp009/016/017/018/035/038/040/045 family). If a future experiment's
  `_input_tensor` changes that order, the flat channel index reported is still correct and
  useful, but the human-readable "which context row / observation slot" label may be off --
  cross-check against that experiment's own `_input_tensor` if the label looks implausible.
- Row-datetime truncation is global across the whole CSV (any row, any location, with
  datetime > T is dropped), which is a superset of "just this row's location" and matches the
  organizers' literal wording ("データのみに切り詰めて") most conservatively.
- This script audits the *input construction* (and, optionally, the deterministic forward pass
  of a fixed checkpoint on that input). It does not re-run training, and it does not verify the
  submission zip's file bytes against a fresh `inference.py` invocation end-to-end -- run
  `inference.py` itself for that; this tool answers the narrower, audit-critical question of
  whether the input tensor for row T could possibly have depended on data timestamped > T.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.util
import inspect
import random
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML is required (pip install pyyaml).", file=sys.stderr)
    raise

try:
    import torch
except ImportError:  # pragma: no cover
    print(
        "PyTorch is required to run the real dataset-loading code (CPU-only build is enough,\n"
        "no CUDA needed). e.g.: pip install torch --index-url https://download.pytorch.org/whl/cpu",
        file=sys.stderr,
    )
    raise


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Filenames observed across this repo's satellite observation files, e.g.
# "test_kanto_region_Himawari_20221231_2330.tif" / "train_aceh_Meteosat_20230101_0000.tif".
OBS_TIMESTAMP_RE = re.compile(r"_(\d{8})_(\d{4})\.tif$")

_LOCAL_MODULE_NAMES = ("dataset", "model", "tiff_utils", "amp_utils", "losses", "normalize_stats")
_MISSING = object()


# --------------------------------------------------------------------------------------
# Experiment module loading: import the *actual* dataset.py (and optionally model.py) from
# the target experiment directory, exactly as if you had `cd`'d into it, without leaking
# stale modules across repeated invocations against different experiment directories.
# --------------------------------------------------------------------------------------


@contextlib.contextmanager
def experiment_import_scope(exp_dir: Path):
    saved = {name: sys.modules.pop(name, _MISSING) for name in _LOCAL_MODULE_NAMES}
    sys.path.insert(0, str(exp_dir))
    try:
        yield
    finally:
        try:
            sys.path.remove(str(exp_dir))
        except ValueError:
            pass
        for name in _LOCAL_MODULE_NAMES:
            sys.modules.pop(name, None)
            if saved[name] is not _MISSING:
                sys.modules[name] = saved[name]


def load_experiment_modules(exp_dir: Path, need_model: bool) -> tuple[Any, Any | None]:
    with experiment_import_scope(exp_dir):
        dataset_module = importlib.import_module("dataset")
        model_module = importlib.import_module("model") if need_model else None
    return dataset_module, model_module


# --------------------------------------------------------------------------------------
# Config / CSV plumbing (mirrors each experiment's own inference.py resolve_path/load_config).
# --------------------------------------------------------------------------------------


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(exp_dir: Path, value: str) -> Path:
    p = Path(value)
    if p.is_absolute():
        return p
    return (exp_dir / p).resolve()


def parse_obs_timestamp(filename: str) -> datetime | None:
    m = OBS_TIMESTAMP_RE.search(filename)
    if not m:
        return None
    date_part, time_part = m.groups()
    return datetime.strptime(date_part + time_part, "%Y%m%d%H%M")


def row_datetime(row: dict[str, str]) -> datetime | None:
    try:
        return datetime.fromisoformat(row["datetime"])
    except (KeyError, ValueError, TypeError):
        return None


def sample_diverse_rows(rows: list[dict[str, str]], num_rows: int, seed: int) -> list[dict[str, str]]:
    """Stratified sample across (name_location, satellite_target) so the audit isn't
    accidentally blind to a satellite or location whose code path differs."""
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        key = (r.get("name_location", ""), r.get("satellite_target", ""))
        groups.setdefault(key, []).append(r)
    rng = random.Random(seed)
    for g in groups.values():
        rng.shuffle(g)
    keys = sorted(groups.keys())
    rng.shuffle(keys)

    selected: list[dict[str, str]] = []
    i = 0
    while len(selected) < num_rows and any(groups[k] for k in keys):
        k = keys[i % len(keys)]
        if groups[k]:
            selected.append(groups[k].pop())
        i += 1
    return selected[:num_rows]


# --------------------------------------------------------------------------------------
# Dataset construction, mirroring each experiment's inference.py call to PrecipDataset(...),
# filtered through inspect so it works whether or not that experiment's constructor accepts
# `features=` (exp038-family) or not (exp016/017/018-family).
# --------------------------------------------------------------------------------------


def build_dataset(dataset_module: Any, rows: list[dict[str, str]], data_dir: Path, config: dict[str, Any]):
    cls = dataset_module.PrecipDataset
    sig = inspect.signature(cls.__init__)
    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    kwargs: dict[str, Any] = dict(
        rows=rows,
        data_dir=data_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=False,
        context_rows=int(config["data"].get("context_rows", 1)),
        norm_stats=None,
        augment=False,
    )
    if "features" in sig.parameters:
        feat_fn = getattr(dataset_module, "features_from_config", None)
        kwargs["features"] = feat_fn(config) if feat_fn else {}
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters}
    return cls(**accepted)


def build_channel_labels(dataset_module: Any, config: dict[str, Any]) -> list[str]:
    """Best-effort flat-channel -> human label map; see module docstring caveat."""
    features_fn = getattr(dataset_module, "features_from_config", None)
    features = features_fn(config) if features_fn else {}
    max_obs = int(config["data"]["max_observations"])
    context_rows = int(config["data"].get("context_rows", 1))
    sat_channels = int(config["data"]["satellite_channels"])

    frame_ch_fn = getattr(dataset_module, "channels_per_frame", None)
    frame_channels = frame_ch_fn(sat_channels, features) if frame_ch_fn else sat_channels
    row_feat_fn = getattr(dataset_module, "channels_per_row", None)
    row_feat_channels = row_feat_fn(features) if row_feat_fn else 0
    satellites = getattr(dataset_module, "SATELLITES", ("goes", "himawari", "meteosat"))

    labels: list[str] = []
    for cr in range(context_rows):
        for obs in range(max_obs):
            for ch in range(frame_channels):
                labels.append(f"map[context_row={cr},obs_slot={obs},ch={ch}]")
    for cr in range(context_rows):
        for obs in range(max_obs):
            labels.append(f"mask[context_row={cr},obs_slot={obs}]")
    if row_feat_channels:
        for cr in range(context_rows):
            for k in range(row_feat_channels):
                labels.append(f"row_feature[context_row={cr},k={k}]")
    for sat in satellites:
        labels.append(f"satellite_onehot[{sat}]")
    return labels


# --------------------------------------------------------------------------------------
# The guarded / logging reader that is swapped in for dataset_module.read_tiff_array.
# --------------------------------------------------------------------------------------


class CausalityViolation(RuntimeError):
    def __init__(self, path: Any, parsed_ts: datetime, target_ts: datetime):
        minutes_past = (parsed_ts - target_ts).total_seconds() / 60.0
        super().__init__(
            f"attempted to read {path!s} timestamped {parsed_ts.isoformat()}, which is "
            f"{minutes_past:.0f} minutes AFTER target time T={target_ts.isoformat()}"
        )
        self.path = str(path)
        self.parsed_ts = parsed_ts
        self.target_ts = target_ts


def make_guarded_reader(real_reader, target_ts: datetime, access_log: list[tuple[str, datetime]], block: bool):
    def guarded(path, *a, **kw):
        ts = parse_obs_timestamp(Path(str(path)).name)
        if ts is not None:
            access_log.append((str(path), ts))
            if block and ts > target_ts:
                raise CausalityViolation(path, ts, target_ts)
        return real_reader(path, *a, **kw)

    return guarded


# --------------------------------------------------------------------------------------
# Per-row audit.
# --------------------------------------------------------------------------------------


@dataclass
class RowAuditResult:
    unique_id: str
    name_location: str
    satellite_target: str
    target_datetime: datetime
    passed: bool
    late_context_rows: list[tuple[int, str, datetime]] = field(default_factory=list)
    late_file_accesses: list[tuple[str, datetime]] = field(default_factory=list)
    truncation_crash: CausalityViolation | None = None
    tensor_shape_mismatch: str | None = None
    tensor_mismatch_channels: list[str] = field(default_factory=list)
    output_mismatch: bool = False

    def reasons(self) -> list[str]:
        reasons = []
        for offset, ctx_id, ctx_dt in self.late_context_rows:
            reasons.append(
                f"context_rows offset={offset} pulls row {ctx_id!r} dated {ctx_dt.isoformat()} "
                f"(> target T) -- this is a successor-row / future-data dependency"
            )
        for path, ts in self.late_file_accesses:
            reasons.append(f"normal-path read {path!r} timestamped {ts.isoformat()} (> target T)")
        if self.truncation_crash is not None:
            reasons.append(f"truncated reconstruction had to reach past T: {self.truncation_crash}")
        if self.tensor_shape_mismatch:
            reasons.append(self.tensor_shape_mismatch)
        if self.tensor_mismatch_channels:
            shown = ", ".join(self.tensor_mismatch_channels[:12])
            more = "" if len(self.tensor_mismatch_channels) <= 12 else f" (+{len(self.tensor_mismatch_channels) - 12} more)"
            reasons.append(f"input tensor differs between normal and truncated construction at: {shown}{more}")
        if self.output_mismatch:
            reasons.append("model output differs between normal and truncated construction (checkpoint forward pass)")
        return reasons


def audit_row(
    dataset_module: Any,
    normal_ds: Any,
    all_rows: list[dict[str, str]],
    data_dir: Path,
    config: dict[str, Any],
    row: dict[str, str],
    real_read_tiff_array,
    channel_labels: list[str] | None,
) -> tuple[RowAuditResult, torch.Tensor, torch.Tensor | None]:
    target_ts = row_datetime(row)
    if target_ts is None:
        raise ValueError(f"row {row.get('unique_id')!r} has an unparsable datetime {row.get('datetime')!r}")

    result = RowAuditResult(
        unique_id=str(row.get("unique_id", "?")),
        name_location=str(row.get("name_location", "?")),
        satellite_target=str(row.get("satellite_target", "?")),
        target_datetime=target_ts,
        passed=True,
    )

    # Signal 1: does the dataset's own _context_rows ever reach a row dated after T?
    context_rows_used = normal_ds._context_rows(row)
    for offset, ctx_row in enumerate(context_rows_used):
        if ctx_row is None:
            continue
        ctx_ts = row_datetime(ctx_row)
        if ctx_ts is not None and ctx_ts > target_ts:
            result.late_context_rows.append((offset, str(ctx_row.get("unique_id", "?")), ctx_ts))

    # Signal 2: normal-path construction with a logging (non-blocking) reader.
    normal_log: list[tuple[str, datetime]] = []
    dataset_module.read_tiff_array = make_guarded_reader(real_read_tiff_array, target_ts, normal_log, block=False)
    try:
        x_normal = normal_ds._input_tensor(row)
    finally:
        dataset_module.read_tiff_array = real_read_tiff_array
    result.late_file_accesses = [(p, ts) for p, ts in normal_log if ts > target_ts]

    # Signal 3: hard-truncation reconstruction + dual-construction tensor identity.
    truncated_rows = [r for r in all_rows if (row_datetime(r) is None) or (row_datetime(r) <= target_ts)]
    truncated_ds = build_dataset(dataset_module, truncated_rows, data_dir, config)
    trunc_log: list[tuple[str, datetime]] = []
    dataset_module.read_tiff_array = make_guarded_reader(real_read_tiff_array, target_ts, trunc_log, block=True)
    x_trunc: torch.Tensor | None = None
    try:
        x_trunc = truncated_ds._input_tensor(row)
    except CausalityViolation as exc:
        result.truncation_crash = exc
    finally:
        dataset_module.read_tiff_array = real_read_tiff_array

    if result.truncation_crash is None and x_trunc is not None:
        if x_normal.shape != x_trunc.shape:
            result.tensor_shape_mismatch = f"shape mismatch: normal={tuple(x_normal.shape)} truncated={tuple(x_trunc.shape)}"
        elif not torch.equal(x_normal, x_trunc):
            diff = x_normal != x_trunc
            flat_diff_any = diff.reshape(diff.shape[0], -1).any(dim=1)
            bad_idx = flat_diff_any.nonzero().flatten().tolist()
            for idx in bad_idx:
                label = channel_labels[idx] if channel_labels and idx < len(channel_labels) else f"channel[{idx}]"
                result.tensor_mismatch_channels.append(label)

    result.passed = (
        not result.late_context_rows
        and not result.late_file_accesses
        and result.truncation_crash is None
        and result.tensor_shape_mismatch is None
        and not result.tensor_mismatch_channels
    )
    return result, x_normal, x_trunc


# --------------------------------------------------------------------------------------
# Optional checkpoint forward-pass comparison (CPU; no CUDA needed).
# --------------------------------------------------------------------------------------


def load_model_cpu(model_module: Any, config: dict[str, Any], checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = model_module.build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.eval()
    return model


@torch.no_grad()
def outputs_match(model_module: Any, model, x_normal: torch.Tensor, x_trunc: torch.Tensor) -> bool:
    pred_fn = model_module.prediction_from_output
    out_normal = pred_fn(model(x_normal.unsqueeze(0)))
    out_trunc = pred_fn(model(x_trunc.unsqueeze(0)))
    return bool(torch.equal(out_normal, out_trunc))


# --------------------------------------------------------------------------------------
# Main.
# --------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", required=True, help="Path to an experiment dir, e.g. g_experiments/exp038")
    parser.add_argument("--config", default="config.yaml", help="Config filename inside --exp-dir (default: config.yaml)")
    parser.add_argument(
        "--csv",
        choices=["evaluation", "train"],
        default="evaluation",
        help="Sample rows from the evaluation CSV (default, matches the audit's own wording -- "
        "'each prediction's target time T') or the train CSV.",
    )
    parser.add_argument("--num-rows", type=int, default=30, help="Number of sampled rows to audit (default 30)")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed (default 0)")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional path (relative to --exp-dir or absolute) to a .pt checkpoint; if given, "
        "also compares the model's forward-pass output between the normal and truncated inputs.",
    )
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        print(f"FATAL: {exp_dir} is not a directory", file=sys.stderr)
        return 2

    config_path = exp_dir / args.config
    if not config_path.exists():
        print(f"FATAL: config {config_path} not found", file=sys.stderr)
        return 2
    config = load_config(config_path)

    need_model = args.checkpoint is not None
    dataset_module, model_module = load_experiment_modules(exp_dir, need_model=need_model)
    real_read_tiff_array = dataset_module.read_tiff_array

    if args.csv == "evaluation":
        csv_path = resolve_path(exp_dir, config["data"]["evaluation_csv"])
        data_dir = resolve_path(exp_dir, config["data"]["evaluation_dir"])
    else:
        csv_path = resolve_path(exp_dir, config["data"]["train_csv"])
        data_dir = resolve_path(exp_dir, config["data"]["train_dir"])

    print(f"[verify_causal_replay] exp_dir       = {exp_dir}")
    print(f"[verify_causal_replay] config        = {config_path}")
    print(f"[verify_causal_replay] csv           = {csv_path}")
    print(f"[verify_causal_replay] data_dir      = {data_dir}")
    print(f"[verify_causal_replay] context_rows  = {config['data'].get('context_rows', 1)}")

    all_rows = dataset_module.read_rows(csv_path)
    sampled = sample_diverse_rows(all_rows, args.num_rows, args.seed)
    print(f"[verify_causal_replay] sampled {len(sampled)} rows spanning "
          f"{len({(r.get('name_location',''), r.get('satellite_target','')) for r in sampled})} (location, satellite) groups")

    normal_ds = build_dataset(dataset_module, all_rows, data_dir, config)
    channel_labels = build_channel_labels(dataset_module, config)

    model = None
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = (exp_dir / ckpt_path).resolve()
        print(f"[verify_causal_replay] checkpoint    = {ckpt_path}")
        model = load_model_cpu(model_module, config, ckpt_path)

    results: list[RowAuditResult] = []
    for row in sampled:
        result, x_normal, x_trunc = audit_row(
            dataset_module, normal_ds, all_rows, data_dir, config, row, real_read_tiff_array, channel_labels
        )
        if result.passed and model is not None and x_trunc is not None:
            try:
                if not outputs_match(model_module, model, x_normal, x_trunc):
                    result.output_mismatch = True
                    result.passed = False
            except Exception as exc:  # pragma: no cover - surfaced in the report either way
                result.output_mismatch = True
                result.passed = False
                result.tensor_mismatch_channels.append(f"(checkpoint forward pass raised: {exc!r})")
        results.append(result)

    print()
    print(f"{'unique_id':<12} {'location':<20} {'sat':<10} {'target_T':<20} {'result'}")
    print("-" * 80)
    n_pass = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        if r.passed:
            n_pass += 1
        print(f"{r.unique_id:<12} {r.name_location:<20} {r.satellite_target:<10} {r.target_datetime.isoformat():<20} {status}")

    print()
    failing = [r for r in results if not r.passed]
    if failing:
        print(f"RESULT: FAIL -- {len(failing)}/{len(results)} sampled rows show a causality dependency on data timestamped > T\n")
        for r in failing:
            print(f"=== {r.unique_id} (location={r.name_location}, satellite={r.satellite_target}, T={r.target_datetime.isoformat()}) ===")
            for reason in r.reasons():
                print(f"  - {reason}")
            print()
        print(
            "This experiment configuration would FAIL the organizers' 2026-07-20 code-inspection "
            "audit (truncate to <= T, re-run, must match) and must not be used for the final submission."
        )
        return 1

    print(f"RESULT: PASS -- all {len(results)} sampled rows are causally clean "
          f"(input tensor identical whether or not data timestamped > T is available; "
          f"no context row or file access reached past T)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
