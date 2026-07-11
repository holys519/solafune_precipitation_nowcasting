#!/usr/bin/env python3
import json, subprocess
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[2]; SRC=ROOT/"g_experiments/exp016"; OUT=ROOT/"outputs/analysis/exp023"; OUT.mkdir(parents=True,exist_ok=True)
rows=[]
for arm,source in (("mean","config.yaml"),("median","config_median_serving.yaml")):
 d=yaml.safe_load((SRC/source).read_text()); d["experiment"]["name"]="exp023"; d["paths"]["model_dir"]=str(ROOT/"g_model/exp016"); d["paths"]["analysis_dir"]=str(OUT/arm); d["paths"]["calibration_path"]=str(OUT/arm/"oof_calibration.json")
 cfg=OUT/arm/"config.yaml"; cfg.parent.mkdir(parents=True,exist_ok=True); cfg.write_text(yaml.safe_dump(d,sort_keys=False))
 subprocess.run(["python3",str(SRC/"analyze_oof.py"),"--config",str(cfg)],check=True)
 s=json.loads((OUT/arm/"analysis_summary.json").read_text()); rows.append({"arm":arm,"oof":s.get("oof_global"),"calibration":s.get("calibration_comparison"),"heavy_rain":s.get("heavy_rain_summary")})
(OUT/"analysis_summary.json").write_text(json.dumps({"arms":rows},indent=2)); print(json.dumps(rows,indent=2))
