#!/usr/bin/env python3
"""Causal-only temporal smoothing for prediction post-processing (g_eda/exp010).

This is an exp010-local copy/extension of `apply_temporal_smoothing` from
`g_experiments/exp038/inference.py` (also carried, untuned, into `g_experiments/exp046/run.py`).
It is deliberately kept as its own copy here -- exp038/exp046's shipped files are not modified by
this experiment, per the project convention of not touching already-submitted green artifacts
from an exploratory OOF sweep. If the recommendation this experiment produces is adopted, a
future harvest experiment (e.g. exp055) should port the winning config into a *new* experiment's
inference/serving code, not edit exp038/exp046 in place.

Extensions over the original:

1. A second causal tap, `prev2_weight` -- the prediction for the same location two steps back
   in time (T-60min, when `max_gap_minutes` per-hop spacing allows it), in addition to the
   existing `prev_weight` tap at T-30min. Together with `center_weight` (T) this gives a
   3-tap-back causal-only window: T, T-30, T-60. `prev2_weight` defaults to 0.0 (pure 2-tap
   behavior, matching the original function exactly).
2. A hard compliance guardrail: when the config sets `causal_only: true`, the function asserts
   `next_weight == 0` and raises loudly (`CausalSmoothingConfigError`) otherwise. The organizers'
   2026-07-20 ruling made non-causal smoothing (mixing a prediction for a target timestamp AFTER
   T into T's prediction) an outright rules violation -- this guard turns a silent
   regression risk (e.g. someone flipping next_weight back on by copy-paste) into an immediate,
   load-bang failure instead of a quiet compliance breach. This is a guard, not just a default:
   it fires even if next_weight is explicitly set to a nonzero value by whoever edits the config.

Weight/renormalization semantics are otherwise unchanged from the original: each tap is only
used if a same-location neighbor exists within `max_gap_minutes` of the required offset (chained
for prev2 -- see `_causal_neighbors` below), and the final blend renormalizes by the sum of
weights actually used (so a tile with no historical neighbor falls back to center_weight-only,
i.e. plain unsmoothed prediction).
"""

from __future__ import annotations

from typing import Any

import numpy as np


class CausalSmoothingConfigError(ValueError):
    """Raised when a `causal_only: true` config actually requests a non-causal tap."""


def _causal_neighbors(
    indices: list[int],
    datetimes: list[np.datetime64],
    pos: int,
    max_gap_minutes: int,
) -> tuple[int | None, int | None]:
    """Return (prev_index, prev2_index) for position `pos` in a location's time-sorted list.

    prev = position pos-1, only if 0 < gap(pos, pos-1) <= max_gap_minutes.
    prev2 = position pos-2, only if prev exists AND 0 < gap(pos-1, pos-2) <= max_gap_minutes
    (i.e. a genuine two-hop causal chain T -> T-30 -> T-60, not merely "some earlier tile").
    """
    prev_idx: int | None = None
    prev2_idx: int | None = None
    if pos > 0:
        gap = (datetimes[pos] - datetimes[pos - 1]) / np.timedelta64(1, "m")
        if 0 < gap <= max_gap_minutes:
            prev_idx = indices[pos - 1]
            if pos > 1:
                gap2 = (datetimes[pos - 1] - datetimes[pos - 2]) / np.timedelta64(1, "m")
                if 0 < gap2 <= max_gap_minutes:
                    prev2_idx = indices[pos - 2]
    return prev_idx, prev2_idx


def apply_temporal_smoothing(items: list[dict[str, Any]], post_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Causal-only temporal smoothing over per-location, time-sorted prediction arrays.

    `items[i]` must have keys: `name_location`, `datetime` (ISO string), `array` (np.ndarray).
    `post_cfg["temporal_smoothing"]` keys: `enabled`, `center_weight`, `prev_weight`,
    `prev2_weight` (new), `next_weight`, `max_gap_minutes`, `causal_only` (new).
    """
    smooth_cfg = post_cfg.get("temporal_smoothing", {})
    if not bool(smooth_cfg.get("enabled", False)):
        return items

    center_weight = float(smooth_cfg.get("center_weight", 0.70))
    prev_weight = float(smooth_cfg.get("prev_weight", 0.15))
    prev2_weight = float(smooth_cfg.get("prev2_weight", 0.0))
    next_weight = float(smooth_cfg.get("next_weight", 0.15))
    max_gap_minutes = int(smooth_cfg.get("max_gap_minutes", 30))

    causal_only = bool(smooth_cfg.get("causal_only", False))
    if causal_only and next_weight != 0.0:
        raise CausalSmoothingConfigError(
            "temporal_smoothing.causal_only=true requires next_weight == 0, got "
            f"next_weight={next_weight!r}. Mixing a later target timestamp's prediction into "
            "T's prediction ('non-causal' smoothing) was ruled a rules violation by the "
            "organizers on 2026-07-20 -- see doc/submission_registry.md."
        )

    by_location: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        by_location.setdefault(str(item["name_location"]), []).append(idx)

    smoothed_arrays: list[np.ndarray | None] = [None] * len(items)
    for indices in by_location.values():
        indices = sorted(indices, key=lambda idx: str(items[idx]["datetime"]))
        datetimes = [np.datetime64(str(items[idx]["datetime"]).replace(" ", "T")) for idx in indices]
        for pos, idx in enumerate(indices):
            weighted = items[idx]["array"] * center_weight
            total_weight = center_weight

            prev_idx, prev2_idx = _causal_neighbors(indices, datetimes, pos, max_gap_minutes)
            if prev_idx is not None and prev_weight > 0:
                weighted = weighted + items[prev_idx]["array"] * prev_weight
                total_weight += prev_weight
            if prev2_idx is not None and prev2_weight > 0:
                weighted = weighted + items[prev2_idx]["array"] * prev2_weight
                total_weight += prev2_weight

            if next_weight > 0 and pos + 1 < len(indices):
                gap = (datetimes[pos + 1] - datetimes[pos]) / np.timedelta64(1, "m")
                if 0 < gap <= max_gap_minutes:
                    weighted = weighted + items[indices[pos + 1]]["array"] * next_weight
                    total_weight += next_weight

            smoothed_arrays[idx] = (weighted / max(total_weight, 1e-8)).astype(np.float32)

    for idx, array in enumerate(smoothed_arrays):
        if array is not None:
            items[idx]["array"] = array
    return items
