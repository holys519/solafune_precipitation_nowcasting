#!/usr/bin/env python3
import argparse
from pathlib import Path
import yaml
p=argparse.ArgumentParser(); p.add_argument("experiment"); p.add_argument("output"); a=p.parse_args()
root=Path(__file__).resolve().parents[2]
d=yaml.safe_load((root/"g_experiments"/a.experiment/"config.yaml").read_text())
d["train"].update({"epochs":50,"early_stopping_patience":10,"early_stopping_min_delta":0.001})
d["scheduler"]={"name":"reduce_on_plateau","factor":0.3,"patience":4,"min_lr":0.00001}
out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(yaml.safe_dump(d,sort_keys=False))
