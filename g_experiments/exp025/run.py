#!/usr/bin/env python3
import argparse, csv, json, subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/"outputs/analysis/exp025"
p=argparse.ArgumentParser()
p.add_argument("--source-exp",default="exp017")
p.add_argument("--source-config",default="config.yaml")
p.add_argument("--seeds",default="42,123,2026")
p.add_argument("--folds",default="0,1,2,3,4")
a=p.parse_args()
src=ROOT/"g_experiments"/a.source_exp
seeds=[int(x) for x in a.seeds.split(",")]
folds=[int(x) for x in a.folds.split(",")]
rows=[]
for seed in seeds:
 d=yaml.safe_load((src/a.source_config).read_text())
 d["experiment"]["name"]="exp025"; d["experiment"]["seed"]=seed
 d["train"].update({"epochs":50,"early_stopping_patience":10,"early_stopping_min_delta":0.001})
 d["scheduler"]={"name":"reduce_on_plateau","factor":0.3,"patience":4,"min_lr":0.00001}
 d["paths"]["model_dir"]=str(ROOT/f"g_model/exp025/seed{seed}")
 d["paths"]["analysis_dir"]=str(OUT/f"seed{seed}")
 cfg=OUT/f"seed{seed}/config.yaml"; cfg.parent.mkdir(parents=True,exist_ok=True)
 cfg.write_text(yaml.safe_dump(d,sort_keys=False))
 for fold in folds:
  subprocess.run(["python3",str(src/"train.py"),"--config",str(cfg),"--fold",str(fold)],check=True)
  m=json.loads((ROOT/f"g_model/exp025/seed{seed}/metrics_fold{fold}.json").read_text())
  rows.append({"source":a.source_exp,"seed":seed,"fold":fold,"best_tile_rmse":m.get("best_tile_rmse"),"best_rmse":m.get("best_rmse")})
OUT.mkdir(parents=True,exist_ok=True)
with (OUT/"seed_fold_summary.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(OUT/"analysis_summary.json").write_text(json.dumps({"runs":rows},indent=2))
print(json.dumps(rows,indent=2))
