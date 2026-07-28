#!/usr/bin/env python3
"""l_eda/exp005 (I-002): submission gate -- combined go/no-go for "should this candidate spend a
submission and/or replace the champion".

This is the operational output of this experiment: `bootstrap_ci.py` answers "is the delta bigger
than fold-composition noise" and `leakage_audit.py` answers "is the gain geography-shaped". Neither
alone would have caught every recent miss on its own -- exp050_sigmafixed's problem was pure noise
(bootstrap would catch it), exp047_sigmafixed's problem was a shortcut riding a real-looking OOF
number (leakage audit would catch it) -- so the gate requires both to pass.

Decision rule (conservative by design -- submissions are scarce with ~10 days left):
    NO-GO  if |point_delta| < NOISE_FLOOR (l_eda/exp003's measured 0.004 residual std)
    NO-GO  if the 80% bootstrap CI does not exclude zero
    HOLD   if both of the above pass but the gain is concentrated (top-2 locations > 60% of it) --
           worth a second look (does the concentration make physical sense, e.g. a genuinely wetter
           test regime) before spending a submission
    GO     if the delta clears the noise floor, the 80% CI excludes zero, and the gain is diffuse

Usage:
    python3 l_eda/exp005/submission_gate.py --baseline exp038_sigmafixed --candidate exp050_sigmafixed
    python3 l_eda/exp005/submission_gate.py --baseline exp038_sigmafixed --candidate exp047_sigmafixed
"""

from __future__ import annotations

import argparse
import json

import bootstrap_ci
import leakage_audit
from oof_common import OUT_DIR


def decide(ci_result: dict, leak_result: dict) -> tuple[str, list[str]]:
    reasons = []
    if ci_result["below_noise_floor"]:
        reasons.append(
            f"point delta {ci_result['point_delta']:+.5f} is below the noise floor "
            f"({ci_result['noise_floor']}) established in l_eda/exp003"
        )
        return "NO-GO", reasons

    if not ci_result["ci80_excludes_zero"]:
        reasons.append(
            f"80% bootstrap CI {ci_result['ci']['80']} includes zero across only "
            f"{ci_result['n_locations']} location clusters -- indistinguishable from "
            "fold-composition noise (see E-4 fold anatomy)"
        )
        return "NO-GO", reasons

    reasons.append(f"point delta {ci_result['point_delta']:+.5f} clears the noise floor")
    reasons.append(f"80% CI {ci_result['ci']['80']} excludes zero "
                    f"(P(candidate better)={ci_result['prob_candidate_better']:.3f})")

    if leak_result["concentration"]["concentrated"]:
        c = leak_result["concentration"]
        reasons.append(
            f"but gain is concentrated: top-2 locations explain "
            f"{c['top2_share_of_positive_gain']:.1%} of the total positive improvement across "
            f"only {c['n_locations_better']}/{c['n_locations']} improved locations -- verify this "
            "isn't a location-identity shortcut (cf. exp047_sigmafixed) before trusting it"
        )
        return "HOLD", reasons

    reasons.append("gain is diffuse across locations -- no geography-shortcut signature detected")
    return "GO", reasons


def run(baseline: str, candidate: str, n_boot: int, seed: int) -> dict:
    ci_result = bootstrap_ci.run(baseline, candidate, n_boot, seed)
    leak_result = leakage_audit.run(baseline, candidate)
    verdict, reasons = decide(ci_result, leak_result)
    return {
        "baseline": baseline,
        "candidate": candidate,
        "verdict": verdict,
        "reasons": reasons,
        "bootstrap_ci": ci_result,
        "leakage_audit": leak_result,
    }


def write_report(result: dict) -> None:
    pair_dir = OUT_DIR / "pairs" / f"{result['candidate']}_vs_{result['baseline']}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    (pair_dir / "submission_gate.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        f"# Submission gate: {result['candidate']} vs {result['baseline']}",
        "",
        f"## Verdict: {result['verdict']}",
        "",
    ]
    for r in result["reasons"]:
        lines.append(f"- {r}")
    lines += ["", "See BOOTSTRAP_CI.md and LEAKAGE_AUDIT.md in this directory for the full detail.", ""]
    (pair_dir / "SUBMISSION_GATE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    bootstrap_ci.write_report(result["bootstrap_ci"])
    leakage_audit.write_report(result["leakage_audit"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = run(args.baseline, args.candidate, args.n_boot, args.seed)
    write_report(result)


if __name__ == "__main__":
    main()
