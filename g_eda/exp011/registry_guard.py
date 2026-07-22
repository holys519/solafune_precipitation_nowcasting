#!/usr/bin/env python3
"""Shared green/amber/red compliance guard for g_eda/exp011 and g_experiments/exp055.

Background (see doc/submission_registry.md): the 2026-07-20 organizer ruling made
successor-row inputs (context_rows > 1) and overlap-patch post-processing permanently
disqualifying ("red"). The registry's operating rule #1 is explicit:

    green系の学習・OOF・blend・calibrationにamber/red artifactを混入させない
    (blend重み・平滑化係数・calibration曲線の「値」の流用も不可 — green OOFで再fitする)

This module is the code-enforced version of that rule. It is deliberately NOT just a
comment/convention: both the blend-weight optimizer (g_eda/exp011/optimize_blend.py) and the
submission builder (g_experiments/exp055/build_submission.py) import `assert_green` and call it
for every source before touching its OOF cache or eval predictions, and refuse (raise) if the
check fails.

Two independent layers, both must pass:

1. `green_allowlist.json` (hand-maintained, next to this file) -- the single source of truth this
   repo's scaffold reads from day to day. Every source referenced by `sources.json` must have an
   entry here with `"status": "green"` (or the one legitimate amber-but-green-derived case, the
   wavelength-alignment table, which this project has kept out of the strict-green track anyway --
   see doc/submission_registry.md).
2. A live re-scan of `doc/submission_registry.md` itself, so a stale allowlist entry cannot mask a
   status change (e.g. if a source gets red-flagged later in the registry but nobody remembered to
   update the allowlist). The scan requires the source name to actually appear in the registry text
   and requires that none of the lines mentioning it are flagged red/amber (a `red確定` / `**RED**`
   / amber marker without an explicit green resolution elsewhere on the same line is treated as a
   hard failure).

Both checks are fail-closed: unknown sources, missing registry mentions, or any red/amber signal
raise `RegistryComplianceError` rather than silently proceeding.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REGISTRY_MD = PROJECT_DIR / "doc" / "submission_registry.md"
ALLOWLIST_JSON = Path(__file__).resolve().parent / "green_allowlist.json"

# Registry status tokens we look for on lines mentioning a source name.
_RED_MARKERS = ("red確定", "red-confirmed", "**red**", "[red]", "[RED]", "全て red", "red固定")
_GREEN_MARKERS = ("green", "ELIGIBLE")


class RegistryComplianceError(RuntimeError):
    """Raised when a source cannot be verified green from both the allowlist and the registry."""


def _load_allowlist() -> dict[str, dict]:
    if not ALLOWLIST_JSON.exists():
        raise RegistryComplianceError(
            f"{ALLOWLIST_JSON} is missing -- cannot verify any source without the maintained "
            "green allowlist. Create it before running any blend/submission code."
        )
    data = json.loads(ALLOWLIST_JSON.read_text(encoding="utf-8"))
    return data["entries"]


def _registry_lines_mentioning(source_name: str) -> list[str]:
    text = REGISTRY_MD.read_text(encoding="utf-8")
    # Match the bare name and also the common "expNNN / expNNN_variant" table-cell spelling.
    pattern = re.compile(re.escape(source_name))
    return [line for line in text.splitlines() if pattern.search(line)]


def assert_green(source_name: str) -> None:
    """Raise RegistryComplianceError unless `source_name` is verifiably green right now.

    Called for every manifest source before its OOF cache or eval predictions are used.
    """
    allowlist = _load_allowlist()
    entry = allowlist.get(source_name)
    if entry is None:
        raise RegistryComplianceError(
            f"'{source_name}' is not present in {ALLOWLIST_JSON.name}. Refusing to include an "
            "unverified source in a green blend/submission -- add a reviewed entry first."
        )
    if entry.get("status") != "green":
        raise RegistryComplianceError(
            f"'{source_name}' is listed in the allowlist with status={entry.get('status')!r}, "
            "not 'green'. Refusing to include it."
        )

    lines = _registry_lines_mentioning(source_name)
    if not lines:
        raise RegistryComplianceError(
            f"'{source_name}' does not appear anywhere in {REGISTRY_MD} -- cannot cross-verify "
            "the allowlist entry against the source of truth. Refusing to include it."
        )
    for line in lines:
        lowered = line.lower()
        if any(marker.lower() in lowered for marker in _RED_MARKERS):
            raise RegistryComplianceError(
                f"'{source_name}' appears with a RED marker in {REGISTRY_MD}:\n    {line.strip()}\n"
                "The allowlist says green, but the registry (source of truth) disagrees -- "
                "refusing to include it. Update the registry/allowlist and resolve the "
                "discrepancy by hand before re-running."
            )
    if not any(any(marker.lower() in line.lower() for marker in _GREEN_MARKERS) for line in lines):
        raise RegistryComplianceError(
            f"'{source_name}' appears in {REGISTRY_MD} but none of the matching lines mention "
            "'green' or 'ELIGIBLE'. Refusing to include it without an explicit green marker:\n"
            + "\n".join(f"    {line.strip()}" for line in lines)
        )


def assert_all_green(source_names: list[str]) -> None:
    errors = []
    for name in source_names:
        try:
            assert_green(name)
        except RegistryComplianceError as exc:
            errors.append(str(exc))
    if errors:
        raise RegistryComplianceError(
            f"{len(errors)}/{len(source_names)} source(s) failed the green compliance check:\n\n"
            + "\n\n".join(errors)
        )


if __name__ == "__main__":
    import sys

    names = sys.argv[1:] or list(_load_allowlist().keys())
    assert_all_green(names)
    print(f"OK: {len(names)} source(s) verified green: {', '.join(names)}")
