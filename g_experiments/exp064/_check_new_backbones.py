#!/usr/bin/env python3
"""Quick pre-flight check: can timm build each new encoder with in_chans=54 and
features_only=True, and does the pretrained_factorized model run a forward+backward pass
without crashing? Run before committing GPU hours to full fold training."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIGS = [
    "config_effb0.yaml",
    "config_effb1.yaml",
    "config_effb2.yaml",
    "config_regnety016.yaml",
    "config_densenet121.yaml",
]


def main() -> None:
    for cfg_file in CONFIGS:
        config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
        in_channels = int(config["model"]["in_channels"])
        try:
            model = build_model(config)
            x = torch.randn(2, in_channels, 41, 41)
            y = torch.rand(2, 1, 41, 41) * (torch.rand(2, 1, 41, 41) > 0.8)
            out = model(x)
            assert out["pred"].shape == (2, 1, 41, 41), out["pred"].shape
            assert (out["pred"] >= 0).all()
            from losses import build_loss

            loss = build_loss(config)(out, y)
            assert torch.isfinite(loss)
            loss.backward()
            params = sum(p.numel() for p in model.parameters())
            print(f"OK  {cfg_file}: encoder={config['model']['encoder_name']} params={params:,} loss={float(loss):.4f}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {cfg_file}: encoder={config['model'].get('encoder_name')}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
