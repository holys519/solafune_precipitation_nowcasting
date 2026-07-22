#!/usr/bin/env python3
"""Cheap static audit: grep an experiment's config(s) and scripts for causality red flags,
and cross-check the result against `doc/submission_registry.md`'s documented green/amber/red
classification for that experiment.

This is the *cheap* companion to `scripts/verify_causal_replay.py` (which actually re-runs the
real dataset-loading code with data truncated to <= T). This script does not import or execute
any project code -- it is a few milliseconds of text parsing, meant as a first-pass tripwire you
run before bothering with the expensive dynamic replay, and as a standing check that the
registry's classification table has not silently drifted out of sync with what a given
experiment's config actually does.

Checks performed, mirroring the organizers' 2026-07-20 ruling
(`doc/submission_registry.md` section "公式裁定・完全版"):

  1. `data.context_rows` (or top-level `context_rows`) > 1
     -> RED: successor-row input (`context_rows: 2` pulls a CSV row dated after the row being
        predicted -- see exp016/017/018/035 in the registry).
  2. Any `overlap`/`patch` module import or invocation in this experiment directory's own
     `inference.py` / `make_submission.py`, OR a same-named script file living in the directory
     (e.g. exp014's `apply_overlap.py`)
     -> RED: overlap-patch reverse engineering (registry: "完全に禁止確定").
  3. `postprocess.temporal_smoothing.enabled: true` with `next_weight > 0`
     -> RED: non-causal (future-direction) prediction smoothing.
     `enabled: true` with `next_weight == 0` is flagged as an INFO note only -- the 2026-07-20
     ruling explicitly permits causal-only (`prev`-only) smoothing.
  4. Cross-reference: every line in `doc/submission_registry.md` that mentions this experiment's
     name is extracted and its declared classification (green/amber/red, by the bold markup used
     in that document) is compared against what checks 1-3 concluded. A mismatch (e.g. the
     registry says green but the config has `context_rows: 2`, or vice versa) is reported as a
     REGISTRY INCONSISTENCY, separate from the red flags above, because either the config or the
     document is out of date and that is itself worth knowing before a final submission decision.

Usage
-----
    python3 scripts/audit_submission_config.py --exp-dir g_experiments/exp038
    python3 scripts/audit_submission_config.py --exp-dir g_experiments/exp038 --config config_features.yaml
    python3 scripts/audit_submission_config.py --exp-dir g_experiments/exp018 --all-configs

Exit code 0 if no red flags are found in the audited config(s) and no registry inconsistency is
detected. Exit code 1 if any red flag is found, or if the audited config's own conclusion
disagrees with what `doc/submission_registry.md` says about this experiment.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    print("PyYAML is required (pip install pyyaml).", file=sys.stderr)
    raise

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "doc" / "submission_registry.md"

OVERLAP_PATCH_NAME_RE = re.compile(r"overlap|patch", re.IGNORECASE)
# Deliberately substring-based, not \b-anchored: exp014's real module is literally named
# "apply_overlap" (overlap preceded by "_", not a word boundary), and future variants may use
# similar compound names (overlap_patch, tile_overlap, patch_predictions, ...). A screening tool
# like this should over-trigger (false positives are a 2-second read) rather than under-trigger
# (a false negative here is a disqualification-grade miss).
OVERLAP_PATCH_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+\S*(?:overlap|patch)\S*",
    re.IGNORECASE | re.MULTILINE,
)
OVERLAP_PATCH_CALL_RE = re.compile(r"\w*(?:overlap|patch)\w*\s*\(", re.IGNORECASE)


@dataclass
class ConfigVerdict:
    config_path: Path
    context_rows: int
    context_rows_flag: bool
    smoothing_enabled: bool
    smoothing_next_weight: float
    smoothing_flag: bool
    overlap_patch_hits: list[str] = field(default_factory=list)

    @property
    def is_red(self) -> bool:
        return self.context_rows_flag or self.smoothing_flag or bool(self.overlap_patch_hits)

    @property
    def static_classification(self) -> str:
        return "red" if self.is_red else "green-or-amber"  # static scan can't distinguish amber (external-spec) reasons


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def check_context_rows(config: dict[str, Any]) -> tuple[int, bool]:
    context_rows = int(config.get("data", {}).get("context_rows", config.get("context_rows", 1)))
    return context_rows, context_rows > 1


def check_temporal_smoothing(config: dict[str, Any]) -> tuple[bool, float, bool]:
    smooth_cfg = config.get("postprocess", {}).get("temporal_smoothing", {}) or {}
    enabled = bool(smooth_cfg.get("enabled", False))
    next_weight = float(smooth_cfg.get("next_weight", 0.0))
    flag = enabled and next_weight > 0
    return enabled, next_weight, flag


def check_overlap_patch(exp_dir: Path) -> list[str]:
    hits: list[str] = []
    for py_file in sorted(exp_dir.glob("*.py")):
        if OVERLAP_PATCH_NAME_RE.search(py_file.stem):
            hits.append(f"{py_file.name}: suspicious filename (contains 'overlap' or 'patch')")
            continue
        try:
            text = py_file.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        lines = text.splitlines()
        for match in OVERLAP_PATCH_IMPORT_RE.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            hits.append(f"{py_file.name}:{line_no}: {lines[line_no - 1].strip()!r}")
        for line_no, line in enumerate(lines, start=1):
            if OVERLAP_PATCH_CALL_RE.search(line) and not OVERLAP_PATCH_IMPORT_RE.match(line):
                hits.append(f"{py_file.name}:{line_no}: {line.strip()!r}")
    for csv_or_other in sorted(exp_dir.glob("*overlap*")) + sorted(exp_dir.glob("*patch*")):
        if csv_or_other.suffix != ".py":
            hits.append(f"{csv_or_other.name}: suspicious artifact filename present in experiment dir")
    return hits


def audit_config(exp_dir: Path, config_path: Path) -> ConfigVerdict:
    config = load_config(config_path)
    context_rows, context_rows_flag = check_context_rows(config)
    smoothing_enabled, next_weight, smoothing_flag = check_temporal_smoothing(config)
    overlap_hits = check_overlap_patch(exp_dir)
    return ConfigVerdict(
        config_path=config_path,
        context_rows=context_rows,
        context_rows_flag=context_rows_flag,
        smoothing_enabled=smoothing_enabled,
        smoothing_next_weight=next_weight,
        smoothing_flag=smoothing_flag,
        overlap_patch_hits=overlap_hits,
    )


# --------------------------------------------------------------------------------------
# Registry cross-reference.
# --------------------------------------------------------------------------------------

BOLD_RED_RE = re.compile(r"\*\*[^*]*red[^*]*\*\*", re.IGNORECASE)
BOLD_GREEN_RE = re.compile(r"\*\*[^*]*green[^*]*\*\*", re.IGNORECASE)
BOLD_AMBER_RE = re.compile(r"\*\*[^*]*amber[^*]*\*\*", re.IGNORECASE)
PLAIN_RED_RE = re.compile(r"\bred\b", re.IGNORECASE)
PLAIN_GREEN_RE = re.compile(r"\bgreen\b", re.IGNORECASE)
PLAIN_AMBER_RE = re.compile(r"\bamber\b", re.IGNORECASE)


def classify_registry_line(line: str) -> str:
    """Best-effort single-word classification of one registry table line/row, preferring bold
    markup (**red (...)** etc, the convention this doc actually uses) and falling back to any
    bare 'red'/'green'/'amber' token. If more than one classification word appears (rare -- only
    seen in the "旧分類" retraction lines), red takes priority since that is the safety-critical
    direction to never miss."""
    if BOLD_RED_RE.search(line):
        return "red"
    if BOLD_AMBER_RE.search(line):
        return "amber"
    if BOLD_GREEN_RE.search(line):
        return "green"
    if PLAIN_RED_RE.search(line):
        return "red"
    if PLAIN_AMBER_RE.search(line):
        return "amber"
    if PLAIN_GREEN_RE.search(line):
        return "green"
    return "unknown"


def find_registry_lines(registry_text: str, exp_name: str) -> list[tuple[str, str]]:
    """Returns [(line, classification), ...] for every registry line mentioning exp_name as a
    whole word (so exp016 doesn't spuriously match exp0165 etc -- not that any exist, but the
    experiment-name suffix families like exp038_sigmafixed do share the exp038 prefix, which is
    exactly the ambiguity this function surfaces rather than hides)."""
    pattern = re.compile(rf"\b{re.escape(exp_name)}\b")
    results = []
    for line in registry_text.splitlines():
        if pattern.search(line) and ("|" in line or "red" in line.lower() or "green" in line.lower() or "amber" in line.lower()):
            if pattern.search(line):
                results.append((line.strip(), classify_registry_line(line)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp-dir", required=True, help="Path to an experiment dir, e.g. g_experiments/exp038")
    parser.add_argument("--config", default=None, help="Specific config filename inside --exp-dir (default: config.yaml)")
    parser.add_argument("--all-configs", action="store_true", help="Audit every config*.yaml in --exp-dir, not just one")
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH), help="Path to doc/submission_registry.md")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_dir():
        print(f"FATAL: {exp_dir} is not a directory", file=sys.stderr)
        return 2

    if args.all_configs:
        config_paths = sorted(exp_dir.glob("config*.yaml"))
    elif args.config:
        config_paths = [exp_dir / args.config]
    else:
        config_paths = [exp_dir / "config.yaml"]

    config_paths = [p for p in config_paths if p.exists()]
    if not config_paths:
        print(f"FATAL: no config found in {exp_dir} (looked for {args.config or 'config.yaml'})", file=sys.stderr)
        return 2

    registry_path = Path(args.registry)
    registry_text = registry_path.read_text(encoding="utf-8") if registry_path.exists() else ""
    exp_name = exp_dir.name
    registry_lines = find_registry_lines(registry_text, exp_name) if registry_text else []

    print(f"[audit_submission_config] exp_dir  = {exp_dir}")
    print(f"[audit_submission_config] registry = {registry_path} ({'found' if registry_text else 'NOT FOUND'})")
    print()

    any_red = False
    any_inconsistency = False

    for config_path in config_paths:
        print(f"=== {config_path.name} ===")
        verdict = audit_config(exp_dir, config_path)

        if verdict.context_rows_flag:
            any_red = True
            print(f"  [RED] data.context_rows = {verdict.context_rows} (> 1) -- successor-row input, "
                  f"forbidden by the 2026-07-20 ruling")
        else:
            print(f"  [ok]  data.context_rows = {verdict.context_rows}")

        if verdict.smoothing_flag:
            any_red = True
            print(f"  [RED] postprocess.temporal_smoothing.enabled=true with next_weight="
                  f"{verdict.smoothing_next_weight} (> 0) -- non-causal (future-direction) smoothing, "
                  f"forbidden by the 2026-07-20 ruling")
        elif verdict.smoothing_enabled:
            print(f"  [info] postprocess.temporal_smoothing.enabled=true but next_weight="
                  f"{verdict.smoothing_next_weight} (causal-only / prev-only) -- explicitly PERMITTED "
                  f"by the 2026-07-20 ruling")
        else:
            print("  [ok]  postprocess.temporal_smoothing disabled")

        if verdict.overlap_patch_hits:
            any_red = True
            print("  [RED] overlap/patch references found in this experiment directory:")
            for hit in verdict.overlap_patch_hits:
                print(f"        - {hit}")
        else:
            print("  [ok]  no overlap/patch references found in this experiment directory")

        if verdict.is_red:
            static_summary = "RED"
        else:
            static_summary = (
                "no red flags (green-or-amber; this script cannot distinguish external-spec-"
                "derived amber reasons -- see doc/submission_registry.md)"
            )
        print(f"  => static verdict for {config_path.name}: {static_summary}")
        print()

    print(f"=== doc/submission_registry.md cross-reference for {exp_name!r} ===")
    if not registry_text:
        print("  [warn] registry file not found or unreadable -- cannot cross-check")
    elif not registry_lines:
        print(f"  [warn] no line in {registry_path.name} mentions {exp_name!r} -- this experiment is "
              f"UNDOCUMENTED in the registry. Add an entry before treating it as a submission candidate.")
        any_inconsistency = True
    else:
        static_says_red = any(v.is_red for v in (audit_config(exp_dir, p) for p in config_paths))
        for line, classification in registry_lines:
            print(f"  registry says [{classification.upper()}]: {line}")
        registry_says_red_anywhere = any(c == "red" for _, c in registry_lines)
        registry_says_green_anywhere = any(c == "green" for _, c in registry_lines)
        if static_says_red and registry_says_green_anywhere and not registry_says_red_anywhere:
            print("  [INCONSISTENCY] static scan found RED-flag settings in this config, but every "
                  "registry line for this experiment says green -- the registry is stale or this "
                  "config file is not the one the registry entry describes. Re-check before trusting "
                  "either.")
            any_inconsistency = True
        elif (not static_says_red) and registry_says_red_anywhere and not registry_says_green_anywhere:
            print("  [INCONSISTENCY] static scan found NO red-flag settings in this config, but the "
                  "registry classifies this experiment red -- likely because the red reason is "
                  "upstream (e.g. blended from red sources) rather than in this config directly; "
                  "confirm against the registry's stated reason column before treating this as green.")
            any_inconsistency = True
        else:
            print("  [consistent] static scan and registry classification agree.")

    print()
    if any_red or any_inconsistency:
        print("RESULT: FAIL -- red flag(s) and/or registry inconsistency found; do not use for final submission "
              "without resolving the item(s) above.")
        return 1
    print("RESULT: PASS -- no red flags found in the audited config(s), and this matches "
          "doc/submission_registry.md's classification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
