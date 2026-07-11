#!/usr/bin/env python3
import argparse, json, subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/"g_experiments/exp017"; OUT=ROOT/"outputs/analysis/exp022"
p=argparse.ArgumentParser(); p.add_argument("--fold",type=int,default=1); a=p.parse_args()
arms={"full":"config.yaml","engineered":"config_engineered_only.yaml","canonical":"config_canonical_only.yaml"}
for arm,name in arms.items():
 d=yaml.safe_load((SRC/name).read_text()); d["experiment"]["name"]="exp022"; d["paths"]["model_dir"]=str(ROOT/"g_model/exp022"/arm); d["paths"]["analysis_dir"]=str(OUT/arm); d["paths"]["output_dir"]=str(ROOT/"outputs/submissions/exp022"/arm)
 d["train"].update({"epochs":50,"early_stopping_patience":10,"early_stopping_min_delta":0.001})
 d["scheduler"]={"name":"reduce_on_plateau","factor":0.3,"patience":4,"min_lr":0.00001}
 cfg=OUT/arm/"config.yaml"; cfg.parent.mkdir(parents=True,exist_ok=True); cfg.write_text(yaml.safe_dump(d,sort_keys=False))
 subprocess.run(["python3",str(SRC/"train.py"),"--config",str(cfg),"--fold",str(a.fold)],check=True)
rows=[]
for arm in arms:
 m=ROOT/"g_model/exp022"/arm/f"metrics_fold{a.fold}.json"; d=json.loads(m.read_text()); rows.append({"arm":arm,"fold":a.fold,"best_tile_rmse":d.get("best_tile_rmse"),"best_rmse":d.get("best_rmse")})
(OUT/f"fold{a.fold}_comparison.json").write_text(json.dumps(rows,indent=2)); print(json.dumps(rows,indent=2))
