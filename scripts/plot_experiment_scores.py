#!/usr/bin/env python3
"""Aggregate historical experiment scores and render dependency-free SVG charts.

The project has several generations of metric JSON schemas.  This script keeps
the source artifacts untouched, normalizes the comparable fields into CSVs,
and makes explicit distinctions between full five-fold OOF, partial-fold CV,
and Public leaderboard results.

Usage:
    python3 scripts/plot_experiment_scores.py
    python3 scripts/plot_experiment_scores.py --output-dir /tmp/score_history
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


FULL_OOF_SAMPLES = 40_686
COLORS = {
    "blue": "#2563eb",
    "green": "#059669",
    "amber": "#d97706",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "slate": "#64748b",
    "light": "#e2e8f0",
    "grid": "#dbe3ee",
    "ink": "#172033",
    "muted": "#64748b",
    "paper": "#ffffff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root (default: inferred from this script)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <root>/doc/score_history)",
    )
    return parser.parse_args()


def experiment_number(value: str) -> int:
    match = re.search(r"exp(\d{3})", value)
    return int(match.group(1)) if match else 10_000


def natural_key(value: str) -> tuple[int, str]:
    return experiment_number(value), value


def experiment_family(value: str) -> str:
    match = re.search(r"exp\d{3}", value)
    if not match:
        raise ValueError(f"No experiment ID in {value!r}")
    return match.group(0)


def as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def analysis_run_label(root: Path, path: Path) -> str:
    rel = path.relative_to(root / "outputs" / "analysis")
    experiment = rel.parts[0]
    nested = list(rel.parts[1:-1])
    suffix = path.stem.removeprefix("analysis_summary").strip("_")
    parts = [experiment, *nested]
    if suffix:
        parts.append(suffix)
    return "/".join(parts)


def normalized_oof_row(
    run: str,
    source_path: Path,
    data: dict[str, Any],
    oof: dict[str, Any],
    *,
    arm: str = "",
) -> dict[str, Any]:
    training = data.get("training") or {}
    selection_metric = data.get("selection_metric") or training.get("selection_metric")
    fold_count = training.get("folds")
    checkpoint_count = data.get("checkpoint_count")
    sample_count = int(oof.get("samples") or 0)
    warnings: list[str] = []
    if fold_count and checkpoint_count and int(fold_count) != int(checkpoint_count):
        warnings.append(f"training_folds={fold_count} but checkpoint_count={checkpoint_count}")
    if sample_count >= FULL_OOF_SAMPLES and checkpoint_count and int(checkpoint_count) < 5:
        warnings.append("full-size OOF assembled from fewer than five reported checkpoints")

    calibration_source = str((data.get("calibration") or {}).get("source") or "")
    full_oof_cutoff = int(FULL_OOF_SAMPLES * 0.99)
    outputs_index = source_path.parts.index("outputs")
    relative_source = Path(*source_path.parts[outputs_index:])
    return {
        "run": f"{run}/{arm}" if arm else run,
        "experiment": experiment_family(run),
        "variant": arm or run.split("/", 1)[1] if "/" in run else arm,
        "coverage": "full_oof" if sample_count >= full_oof_cutoff else "partial_oof",
        "samples": sample_count,
        "fold_count": fold_count,
        "checkpoint_count": checkpoint_count,
        "selection_metric": selection_metric,
        "cv_metric_mean": as_float(
            training.get("best_metric_mean", training.get("best_rmse_mean"))
        ),
        "cv_metric_std": as_float(
            training.get("best_metric_std", training.get("best_rmse_std"))
        ),
        "oof_tile_rmse": as_float(oof.get("tile_rmse", data.get("oof_official_metric"))),
        "oof_global_rmse": as_float(oof.get("rmse")),
        "oof_mae": as_float(oof.get("mae")),
        "oof_bias": as_float(oof.get("bias")),
        "best_rain_threshold_tile_rmse": as_float(
            (data.get("best_rain_threshold") or {}).get("tile_rmse")
        ),
        "best_value_threshold_tile_rmse": as_float(
            (data.get("best_value_threshold") or {}).get("tile_rmse")
        ),
        "calibration_source": calibration_source,
        "score_source": "analysis_summary",
        "artifact_warning": "; ".join(warnings),
        "source_path": str(relative_source),
    }


def load_cv_runs(root: Path) -> list[dict[str, Any]]:
    analysis_root = root / "outputs" / "analysis"
    rows: list[dict[str, Any]] = []
    for path in sorted(analysis_root.glob("**/analysis_summary*.json")):
        data = read_json(path)
        run = analysis_run_label(root, path)
        oof = data.get("oof_global")
        if isinstance(oof, dict):
            rows.append(normalized_oof_row(run, path, data, oof))
        for arm in data.get("arms") or []:
            arm_oof = arm.get("oof")
            if isinstance(arm_oof, dict):
                rows.append(
                    normalized_oof_row(
                        run,
                        path,
                        data,
                        arm_oof,
                        arm=str(arm.get("arm") or "unnamed"),
                    )
                )

    # exp003-exp006 predate tile_rmse in analysis_summary.json.  The E-3 audit
    # recomputed it from their OOF predictions; merge that durable audit value.
    calibration_csv = root / "outputs" / "l_eda" / "exp003" / "cv_lb_pairs.csv"
    backfill: dict[str, float] = {}
    if calibration_csv.exists():
        with calibration_csv.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                value = as_float(row.get("oof_tile_rmse"))
                if value is not None:
                    backfill[row["experiment"]] = value
    for row in rows:
        if row["oof_tile_rmse"] is None and row["experiment"] in backfill:
            row["oof_tile_rmse"] = backfill[row["experiment"]]
            row["score_source"] = "E-3 OOF recomputation"

    # Some experiments retain both a parent summary with embedded arms and a
    # more detailed child summary for each arm. Prefer the row with richer
    # training metadata rather than plotting the same OOF prediction twice.
    deduplicated: dict[tuple[str, float | None, float | None], dict[str, Any]] = {}
    for row in rows:
        key = (row["run"], row["oof_tile_rmse"], row["oof_global_rmse"])
        previous = deduplicated.get(key)
        richness = sum(row.get(field) not in {None, ""} for field in ("fold_count", "checkpoint_count", "cv_metric_mean"))
        previous_richness = -1 if previous is None else sum(
            previous.get(field) not in {None, ""}
            for field in ("fold_count", "checkpoint_count", "cv_metric_mean")
        )
        if previous is None or richness > previous_richness:
            deduplicated[key] = row

    return sorted(deduplicated.values(), key=lambda row: natural_key(row["run"]))


def infer_fold_from_path(path: Path, data: dict[str, Any]) -> int | None:
    if data.get("fold") is not None:
        return int(data["fold"])
    match = re.search(r"fold(\d+)", path.stem)
    return int(match.group(1)) if match else None


def load_fold_scores(root: Path) -> list[dict[str, Any]]:
    model_root = root / "g_model"
    rows: list[dict[str, Any]] = []
    if not model_root.exists():
        return rows
    for path in sorted(model_root.glob("**/metrics*.json")):
        data = read_json(path)
        fold = infer_fold_from_path(path, data)
        rel_parent = path.parent.relative_to(model_root)
        run = "/".join(rel_parent.parts)
        tile = as_float(data.get("best_tile_rmse"))
        rmse = as_float(data.get("best_rmse"))
        rows.append(
            {
                "run": run,
                "experiment": experiment_family(run),
                "fold": fold,
                "selection_metric": str(data.get("selection_metric") or "rmse"),
                "best_tile_rmse": tile,
                "best_rmse": rmse,
                "selected_score": tile if tile is not None else rmse,
                "valid_rows": data.get("valid_rows_used"),
                "valid_locations": ";".join(data.get("valid_locations") or []),
                "epochs_completed": data.get("epochs_completed", len(data.get("history") or [])),
                "source_path": str(path.relative_to(root)),
            }
        )
    return sorted(rows, key=lambda row: (*natural_key(row["run"]), row["fold"] or -1))


def public_risk(experiment: str, submission: str) -> str:
    family = experiment_family(experiment)
    if "patched" in submission or family in {"exp014", "exp026"}:
        return "red"
    if family in {
        "exp009",
        "exp015",
        "exp016",
        "exp017",
        "exp018",
        "exp024",
        "exp027",
        "exp033",
        "exp035",
        "exp036",
        "exp037",
        "exp039",
    }:
        return "amber"
    return "green"


def load_public_scores(root: Path) -> list[dict[str, Any]]:
    path = root / "doc" / "public_scores.md"
    text = path.read_text(encoding="utf-8")
    section = text.split("## Submission Log", 1)[1].split("## Leaderboard Context", 1)[0]
    rows: list[dict[str, Any]] = []
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 7 or cells[0] in {"Submitted at", "---"} or set(cells[0]) <= {"-", ":"}:
            continue
        try:
            submitted = datetime.strptime(cells[0], "%Y/%m/%d %H:%M:%S")
            score = float(cells[3])
        except ValueError:
            continue
        submission = cells[2].strip("`")
        rows.append(
            {
                "submitted_at": submitted.isoformat(sep=" "),
                "run": cells[1],
                "experiment": experiment_family(cells[1]),
                "submission": submission,
                "public_rmse": score,
                "status": cells[5],
                "risk": public_risk(cells[1], submission),
                "memo": cells[6].replace("**", "").replace("`", ""),
                "source": "doc/public_scores.md#submission-log",
            }
        )

    # E-3 contains one submitted pure-model score that has not yet been copied
    # into public_scores.md. Preserve it in the per-experiment view, but do not
    # invent a timestamp, so it is intentionally absent from the timeline.
    e3_path = root / "outputs" / "l_eda" / "exp003" / "cv_lb_pairs.csv"
    known = {(row["run"], round(row["public_rmse"], 12)) for row in rows}
    if e3_path.exists():
        with e3_path.open(newline="", encoding="utf-8") as handle:
            for e3 in csv.DictReader(handle):
                score = as_float(e3.get("public_rmse"))
                key = (e3["experiment"], round(score, 12) if score is not None else None)
                if score is None or key in known:
                    continue
                rows.append(
                    {
                        "submitted_at": "",
                        "run": e3["experiment"],
                        "experiment": experiment_family(e3["experiment"]),
                        "submission": "(recorded in E-3 audit)",
                        "public_rmse": score,
                        "status": "valid",
                        "risk": public_risk(e3["experiment"], ""),
                        "memo": "Timestamp/submission name missing from public_scores.md",
                        "source": "outputs/l_eda/exp003/cv_lb_pairs.csv",
                    }
                )
    return sorted(rows, key=lambda row: (row["submitted_at"] or "9999", row["run"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def svg_open(width: int, height: int, title: str, subtitle: str = "") -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Inter, DejaVu Sans, Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #64748b; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; fill: #64748b; }",
        ".value { font-size: 12px; font-weight: 600; }",
        "</style>",
        f'<rect width="{width}" height="{height}" fill="{COLORS["paper"]}"/>',
        f'<text x="28" y="36" class="title">{escape(title)}</text>',
        f'<text x="28" y="58" class="subtitle">{escape(subtitle)}</text>',
    ]


def scale(value: float, low: float, high: float, start: float, end: float) -> float:
    if high <= low:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def tick_values(low: float, high: float, count: int = 6) -> list[float]:
    if high <= low:
        return [low]
    raw = (high - low) / max(count - 1, 1)
    magnitude = 10 ** math.floor(math.log10(raw))
    normalized = raw / magnitude
    step_base = 1 if normalized <= 1 else 2 if normalized <= 2 else 5 if normalized <= 5 else 10
    step = step_base * magnitude
    first = math.floor(low / step) * step
    ticks: list[float] = []
    current = first
    while current <= high + step:
        if current >= low - 1e-12:
            ticks.append(current)
        current += step
    return ticks


def write_dot_chart(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    title: str,
    subtitle: str,
    label_key: str,
    value_key: str,
    color_key: str | None = None,
    color_map: dict[str, str] | None = None,
    note_key: str | None = None,
) -> None:
    rows = [row for row in rows if as_float(row.get(value_key)) is not None]
    row_height = 29
    width = 1240
    top, bottom, left, right = 92, 62, 250, 145
    height = top + bottom + row_height * max(len(rows), 1)
    values = [float(row[value_key]) for row in rows]
    padding = max((max(values) - min(values)) * 0.08, 0.002)
    low, high = min(values) - padding, max(values) + padding
    chart_right = width - right
    svg = svg_open(width, height, title, subtitle)
    for tick in tick_values(low, high):
        x = scale(tick, low, high, left, chart_right)
        svg.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="{COLORS["grid"]}"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-28}" text-anchor="middle" class="small">{tick:.3f}</text>')
    svg.append(f'<text x="{chart_right}" y="{height-9}" text-anchor="end" class="small">RMSE (lower is better)</text>')
    for index, row in enumerate(rows):
        y = top + index * row_height + row_height / 2
        value = float(row[value_key])
        x = scale(value, low, high, left, chart_right)
        label = str(row[label_key])
        color_value = str(row.get(color_key) or "blue") if color_key else "blue"
        color = (color_map or {}).get(color_value, COLORS.get(color_value, COLORS["blue"]))
        if index % 2:
            svg.append(f'<rect x="18" y="{y-row_height/2:.1f}" width="{width-36}" height="{row_height}" fill="#f8fafc"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="label">{escape(label)}</text>')
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}"/>')
        svg.append(f'<text x="{chart_right+12}" y="{y+4:.1f}" class="value">{value:.6f}</text>')
        if note_key and row.get(note_key):
            svg.append(f'<title>{escape(str(row[note_key]))}</title>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_fold_chart(path: Path, fold_rows: list[dict[str, Any]]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in fold_rows:
        if row["best_tile_rmse"] is not None:
            grouped[row["run"]].append(float(row["best_tile_rmse"]))
    series = [(run, values) for run, values in grouped.items() if len(values) >= 2]
    series.sort(key=lambda item: natural_key(item[0]))
    width, row_height = 1240, 31
    top, bottom, left, right = 92, 64, 250, 120
    height = top + bottom + row_height * max(len(series), 1)
    all_values = [value for _, values in series for value in values]
    low, high = min(all_values) - 0.03, max(all_values) + 0.03
    chart_right = width - right
    svg = svg_open(
        width,
        height,
        "Fold variability of tile RMSE",
        "Dots are held-out-location folds; diamond is the unweighted fold mean. Large spread is mainly geography/rain-regime shift.",
    )
    for tick in tick_values(low, high, 8):
        x = scale(tick, low, high, left, chart_right)
        svg.append(f'<line x1="{x:.1f}" y1="{top-8}" x2="{x:.1f}" y2="{height-bottom+5}" stroke="{COLORS["grid"]}"/>')
        svg.append(f'<text x="{x:.1f}" y="{height-29}" text-anchor="middle" class="small">{tick:.2f}</text>')
    for index, (run, values) in enumerate(series):
        y = top + index * row_height + row_height / 2
        xs = [scale(value, low, high, left, chart_right) for value in values]
        if index % 2:
            svg.append(f'<rect x="18" y="{y-row_height/2:.1f}" width="{width-36}" height="{row_height}" fill="#f8fafc"/>')
        svg.append(f'<text x="{left-12}" y="{y+4:.1f}" text-anchor="end" class="label">{escape(run)}</text>')
        svg.append(f'<line x1="{min(xs):.1f}" y1="{y:.1f}" x2="{max(xs):.1f}" y2="{y:.1f}" stroke="{COLORS["slate"]}" stroke-width="2"/>')
        for x in xs:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.2" fill="{COLORS["blue"]}" opacity="0.75"/>')
        mean_x = scale(mean(values), low, high, left, chart_right)
        svg.append(f'<path d="M {mean_x:.1f} {y-7:.1f} L {mean_x+7:.1f} {y:.1f} L {mean_x:.1f} {y+7:.1f} L {mean_x-7:.1f} {y:.1f} Z" fill="{COLORS["red"]}"/>')
        svg.append(f'<text x="{chart_right+12}" y="{y+4:.1f}" class="value">{mean(values):.4f} ± {pstdev(values):.4f}</text>')
    svg.append(f'<text x="{chart_right}" y="{height-9}" text-anchor="end" class="small">tile RMSE (lower is better)</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def write_public_timeline(path: Path, public_rows: list[dict[str, Any]]) -> None:
    rows = [row for row in public_rows if row["submitted_at"]]
    rows.sort(key=lambda row: row["submitted_at"])
    width, height = 1360, 720
    left, right, top, bottom = 82, 42, 92, 100
    chart_right, chart_bottom = width - right, height - bottom
    scores = [float(row["public_rmse"]) for row in rows]
    low, high = min(scores) - 0.006, max(scores) + 0.006
    svg = svg_open(
        width,
        height,
        "Public leaderboard submission history",
        "Every recorded valid submission; red line is best-so-far. Risk colors: green / amber / red under the current rules audit.",
    )
    for tick in tick_values(low, high, 8):
        y = scale(tick, low, high, top, chart_bottom)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick:.3f}</text>')
    best = math.inf
    best_points: list[tuple[float, float]] = []
    new_best_indices: set[int] = set()
    for index, row in enumerate(rows):
        x = scale(index, 0, max(len(rows) - 1, 1), left, chart_right)
        value = float(row["public_rmse"])
        if value < best:
            best = value
            new_best_indices.add(index)
        best_points.append((x, scale(best, low, high, top, chart_bottom)))
    svg.append('<polyline points="{}" fill="none" stroke="{}" stroke-width="2.5"/>'.format(
        " ".join(f"{x:.1f},{y:.1f}" for x, y in best_points), COLORS["red"]
    ))
    last_label_y = -999.0
    for index, row in enumerate(rows):
        x = scale(index, 0, max(len(rows) - 1, 1), left, chart_right)
        y = scale(float(row["public_rmse"]), low, high, top, chart_bottom)
        color = COLORS.get(row["risk"], COLORS["slate"])
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"><title>{escape(row["run"] + " " + row["submission"] + " " + str(row["public_rmse"]))}</title></circle>')
        if index in new_best_indices and abs(y - last_label_y) > 19:
            svg.append(f'<text x="{x+5:.1f}" y="{y-9:.1f}" class="small">{escape(row["run"])} {float(row["public_rmse"]):.4f}</text>')
            last_label_y = y
        if index % 4 == 0 or index == len(rows) - 1:
            date = row["submitted_at"][:10]
            svg.append(f'<text x="{x:.1f}" y="{chart_bottom+25}" text-anchor="end" transform="rotate(-42 {x:.1f} {chart_bottom+25})" class="small">{escape(date)}</text>')
    svg.append(f'<text x="{left}" y="{top-14}" class="small">lower is better ↑</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def load_cv_lb_pairs(root: Path) -> list[dict[str, Any]]:
    path = root / "outputs" / "l_eda" / "exp003" / "cv_lb_pairs.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_cv_lb_scatter(root: Path, path: Path, pairs: list[dict[str, Any]]) -> None:
    if not pairs:
        return
    stats_path = root / "outputs" / "l_eda" / "exp003" / "calibration_stats.json"
    stats = read_json(stats_path).get("oof_tile_rmse", {}) if stats_path.exists() else {}
    slope = float(stats.get("slope", 1.0))
    intercept = float(stats.get("intercept", 0.0))
    width, height = 920, 760
    left, right, top, bottom = 90, 50, 96, 82
    chart_right, chart_bottom = width - right, height - bottom
    xs = [float(row["oof_tile_rmse"]) for row in pairs]
    ys = [float(row["public_rmse"]) for row in pairs]
    xlow, xhigh = min(xs) - 0.004, max(xs) + 0.004
    ylow, yhigh = min(ys) - 0.004, max(ys) + 0.004
    svg = svg_open(
        width,
        height,
        "Five-fold OOF vs Public LB",
        f"Historical E-3 audit; all-pair Spearman={float(stats.get('spearman', float('nan'))):.3f}, pure-model Spearman={float((stats.get('pure_model') or {}).get('spearman', float('nan'))):.3f}.",
    )
    for tick in tick_values(xlow, xhigh, 7):
        x = scale(tick, xlow, xhigh, left, chart_right)
        svg.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{chart_bottom}" stroke="{COLORS["grid"]}"/>')
        svg.append(f'<text x="{x:.1f}" y="{chart_bottom+24}" text-anchor="middle" class="small">{tick:.3f}</text>')
    for tick in tick_values(ylow, yhigh, 7):
        y = scale(tick, ylow, yhigh, chart_bottom, top)
        svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{chart_right}" y2="{y:.1f}" stroke="{COLORS["grid"]}"/>')
        svg.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" class="small">{tick:.3f}</text>')
    x1, x2 = xlow, xhigh
    y1, y2 = slope * x1 + intercept, slope * x2 + intercept
    svg.append(f'<line x1="{scale(x1,xlow,xhigh,left,chart_right):.1f}" y1="{scale(y1,ylow,yhigh,chart_bottom,top):.1f}" x2="{scale(x2,xlow,xhigh,left,chart_right):.1f}" y2="{scale(y2,ylow,yhigh,chart_bottom,top):.1f}" stroke="{COLORS["red"]}" stroke-width="2"/>')
    for row in pairs:
        x_value, y_value = float(row["oof_tile_rmse"]), float(row["public_rmse"])
        x = scale(x_value, xlow, xhigh, left, chart_right)
        y = scale(y_value, ylow, yhigh, chart_bottom, top)
        is_model = row["kind"] == "model"
        color = COLORS["blue"] if is_model else COLORS["amber"]
        if is_model:
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        else:
            svg.append(f'<path d="M {x-6:.1f} {y-6:.1f} L {x+6:.1f} {y+6:.1f} M {x+6:.1f} {y-6:.1f} L {x-6:.1f} {y+6:.1f}" stroke="{color}" stroke-width="2.5"/>')
        svg.append(f'<text x="{x+8:.1f}" y="{y-7:.1f}" class="small">{escape(row["experiment"])}</text>')
    svg.append(f'<text x="{(left+chart_right)/2:.1f}" y="{height-18}" text-anchor="middle" class="label">5-fold OOF tile RMSE</text>')
    svg.append(f'<text x="20" y="{(top+chart_bottom)/2:.1f}" text-anchor="middle" transform="rotate(-90 20 {(top+chart_bottom)/2:.1f})" class="label">Public RMSE</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def make_coverage(
    root: Path,
    cv_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    experiments = {
        path.name
        for path in (root / "g_experiments").glob("exp[0-9][0-9][0-9]")
        if path.is_dir()
    }
    experiments.update(row["experiment"] for row in cv_rows)
    experiments.update(row["experiment"] for row in fold_rows)
    experiments.update(row["experiment"] for row in public_rows)
    result: list[dict[str, Any]] = []
    for experiment in sorted(experiments, key=natural_key):
        cv = [row for row in cv_rows if row["experiment"] == experiment]
        folds = [row for row in fold_rows if row["experiment"] == experiment]
        public = [row for row in public_rows if row["experiment"] == experiment]
        full = [row for row in cv if row["coverage"] == "full_oof" and row["oof_tile_rmse"] is not None]
        partial = [row for row in cv if row["coverage"] == "partial_oof"]
        if full:
            status = "full_oof"
        elif folds or partial:
            status = "partial_cv_only"
        elif public:
            status = "public_only"
        else:
            status = "no_score_artifact"
        result.append(
            {
                "experiment": experiment,
                "status": status,
                "full_oof_runs": ";".join(row["run"] for row in full),
                "partial_cv_runs": ";".join(sorted({row["run"] for row in partial})),
                "metric_json_count": len(folds),
                "public_submission_count": len(public),
                "best_public_rmse": min((float(row["public_rmse"]) for row in public), default=""),
            }
        )
    return result


def write_report(
    root: Path,
    output_dir: Path,
    cv_rows: list[dict[str, Any]],
    fold_rows: list[dict[str, Any]],
    public_rows: list[dict[str, Any]],
    coverage: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> None:
    full = [row for row in cv_rows if row["coverage"] == "full_oof" and row["oof_tile_rmse"] is not None]
    full.sort(key=lambda row: float(row["oof_tile_rmse"]))
    public_best: dict[str, dict[str, Any]] = {}
    for row in public_rows:
        if row["experiment"] not in public_best or float(row["public_rmse"]) < float(public_best[row["experiment"]]["public_rmse"]):
            public_best[row["experiment"]] = row
    exp038 = next((row for row in full if row["run"] == "exp038"), None)
    exp011 = next((row for row in full if row["run"] == "exp011"), None)
    exp038_public = public_best.get("exp038")
    stats_path = root / "outputs" / "l_eda" / "exp003" / "calibration_stats.json"
    calibration = read_json(stats_path).get("oof_tile_rmse", {}) if stats_path.exists() else {}
    predicted = None
    if exp038 and calibration:
        predicted = float(calibration["slope"]) * float(exp038["oof_tile_rmse"]) + float(calibration["intercept"])

    lines = [
        "# Experiment score history",
        "",
        f"Generated {datetime.now(timezone.utc).date().isoformat()} from local JSON/CSV artifacts and `doc/public_scores.md`.",
        "",
        "## CV readiness verdict",
        "",
        "The project can use full five-fold, location-held-out OOF `tile_rmse` for Public-LB-free model selection, especially for changes larger than the historical calibration residual scale. It is not yet strong enough to treat very small deltas as conclusive.",
        "",
        f"- E-3 matched pure-model pairs: {sum(row.get('kind') == 'model' for row in pairs)}; five-fold OOF/Public Spearman = {float((calibration.get('pure_model') or {}).get('spearman', float('nan'))):.3f}.",
        f"- In-sample pure-model calibration residual std = {float((calibration.get('pure_model') or {}).get('resid_std', float('nan'))):.4f}; differences below roughly this scale need paired fold/location or outer-CV evidence.",
        "- The split is leakage-resistant by `name_location`, but only about 20 train locations / 28 location-month blocks are effectively independent, so fold climate variance is large.",
        "- The hidden metric aggregation remains ambiguous: official material suggests pooled RMSE, while historical Public LB is much better explained by mean per-tile RMSE.",
        "- Ensemble/postprocess experiments without their own cross-fitted OOF cannot be judged from CV alone; reusing an upstream model's OOF is not a matched evaluation.",
        "",
        "## exp038 strict-green result",
        "",
    ]
    if exp038:
        lines.append(
            f"- Full five-fold OOF tile RMSE: **{float(exp038['oof_tile_rmse']):.6f}**; global pooled RMSE: {float(exp038['oof_global_rmse']):.6f}; fold-best mean/std: {float(exp038['cv_metric_mean']):.6f} ± {float(exp038['cv_metric_std']):.6f}."
        )
        if exp038["best_rain_threshold_tile_rmse"] is not None:
            threshold_delta = float(exp038["best_rain_threshold_tile_rmse"]) - float(exp038["oof_tile_rmse"])
            lines.append(
                f"- OOF-selected rain threshold reaches {float(exp038['best_rain_threshold_tile_rmse']):.6f} ({threshold_delta:+.6f}); this delta is too small to treat as robust by the historical 0.0041 scale."
            )
    if exp038 and exp011:
        delta = float(exp038["oof_tile_rmse"]) - float(exp011["oof_tile_rmse"])
        lines.append(f"- Versus strict-green exp011 ({float(exp011['oof_tile_rmse']):.6f}): **{delta:+.6f}** OOF improvement.")
    if exp038_public is not None:
        actual = float(exp038_public["public_rmse"])
        lines.append(f"- Submitted Public RMSE: **{actual:.6f}** (valid; current strict/green champion).")
    if predicted is not None and exp038_public is not None:
        actual = float(exp038_public["public_rmse"])
        lines.append(
            f"- The pre-submission historical E-3 mapping predicted **{predicted:.4f}**; actual was {actual:.4f} ({actual - predicted:+.4f}). This single residual is post-hoc and is not a new calibration target."
        )
    elif predicted is not None:
        lines.append(
            f"- Historical E-3 linear mapping gives an indicative Public RMSE of **{predicted:.4f}**. This is a model-family interpolation, not a leaderboard result or a confidence interval."
        )
    lines.extend([
        "",
        "## Best full-OOF runs",
        "",
        "| Run | Coverage | OOF tile RMSE | Global RMSE | Warning |",
        "| --- | --- | ---: | ---: | --- |",
    ])
    for row in full:
        global_rmse = "" if row["oof_global_rmse"] is None else f"{float(row['oof_global_rmse']):.6f}"
        lines.append(
            f"| {row['run']} | {row['coverage']} | {float(row['oof_tile_rmse']):.6f} | {global_rmse} | {row['artifact_warning']} |"
        )
    lines.extend([
        "",
        "## Coverage",
        "",
        f"- Normalized OOF rows: {len(cv_rows)} ({len(full)} full-size tile-RMSE rows).",
        f"- Fold metric rows: {len(fold_rows)}.",
        f"- Public records: {len(public_rows)} across {len(public_best)} experiments.",
        "- Experiments with no score artifact: " + ", ".join(row["experiment"] for row in coverage if row["status"] == "no_score_artifact") + ".",
        "",
        "## Files",
        "",
        "- `cv_oof_by_experiment.svg`: comparable full-size OOF tile RMSE.",
        "- `fold_variability.svg`: per-fold held-out-location variability.",
        "- `public_best_by_experiment.svg`: best recorded Public score per experiment.",
        "- `public_lb_timeline.svg`: all timestamped submissions and best-so-far.",
        "- `cv_vs_public.svg`: audited E-3 matched CV/Public pairs.",
        "- CSV files are the normalized source tables behind the charts.",
        "- Regenerate with `python3 scripts/plot_experiment_scores.py` (standard library only).",
        "",
        "## Source caveats",
        "",
        "- `outputs/` and model artifacts are git-ignored; this directory is a durable snapshot, but rerunning requires the local artifacts to still exist.",
        "- `doc/public_scores.md` is hand-maintained. The E-3-only exp035_no_dilation score is included in the per-experiment chart without inventing a timestamp.",
        "- exp003-exp006 OOF tile RMSE values are backfilled from the E-3 recomputation because their original summary schema stored only pooled RMSE.",
        "- `artifact_warning` flags schema/provenance inconsistencies such as exp030 reporting one training metric file but five OOF checkpoints.",
        "",
    ])
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = (args.output_dir or (root / "doc" / "score_history")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cv_rows = load_cv_runs(root)
    fold_rows = load_fold_scores(root)
    public_rows = load_public_scores(root)
    pairs = load_cv_lb_pairs(root)
    coverage = make_coverage(root, cv_rows, fold_rows, public_rows)

    write_csv(output_dir / "cv_runs.csv", cv_rows)
    write_csv(output_dir / "fold_scores.csv", fold_rows)
    write_csv(output_dir / "public_submissions.csv", public_rows)
    write_csv(output_dir / "experiment_coverage.csv", coverage)

    full_oof = [
        row
        for row in cv_rows
        if row["coverage"] == "full_oof" and row["oof_tile_rmse"] is not None
    ]
    write_dot_chart(
        output_dir / "cv_oof_by_experiment.svg",
        full_oof,
        title="Full-size OOF tile RMSE by experiment",
        subtitle="Near-complete location-held-out OOF (>=99% of 40,686 rows). Older schemas are backfilled from the E-3 audit.",
        label_key="run",
        value_key="oof_tile_rmse",
        color_key="score_source",
        color_map={"analysis_summary": COLORS["blue"], "E-3 OOF recomputation": COLORS["purple"]},
        note_key="artifact_warning",
    )
    write_fold_chart(output_dir / "fold_variability.svg", fold_rows)

    best_public: list[dict[str, Any]] = []
    for experiment in sorted({row["experiment"] for row in public_rows}, key=natural_key):
        choices = [row for row in public_rows if row["experiment"] == experiment]
        best_public.append(min(choices, key=lambda row: float(row["public_rmse"])))
    write_dot_chart(
        output_dir / "public_best_by_experiment.svg",
        best_public,
        title="Best recorded Public RMSE by experiment",
        subtitle="One best submission per experiment; colors follow the current green / amber / red rules audit, not submission validity.",
        label_key="run",
        value_key="public_rmse",
        color_key="risk",
        color_map={key: COLORS[key] for key in ("green", "amber", "red")},
        note_key="submission",
    )
    write_public_timeline(output_dir / "public_lb_timeline.svg", public_rows)
    write_cv_lb_scatter(root, output_dir / "cv_vs_public.svg", pairs)
    write_report(root, output_dir, cv_rows, fold_rows, public_rows, coverage, pairs)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "full_oof_samples": FULL_OOF_SAMPLES,
        "cv_rows": len(cv_rows),
        "fold_rows": len(fold_rows),
        "public_rows": len(public_rows),
        "experiments": len(coverage),
        "sources": [
            "outputs/analysis/**/analysis_summary*.json",
            "g_model/**/metrics*.json",
            "doc/public_scores.md#Submission-Log",
            "outputs/l_eda/exp003/cv_lb_pairs.csv",
            "outputs/l_eda/exp003/calibration_stats.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote score tables, report, and SVG charts to {output_dir}")


if __name__ == "__main__":
    main()
