#!/usr/bin/env python3
import csv, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
out = ROOT / "outputs/analysis/exp021"; out.mkdir(parents=True, exist_ok=True)
rows=[]
for exp in ("exp009", "exp016", "exp017"):
    p=ROOT/"outputs/analysis"/exp/"analysis_summary.json"
    if not p.exists(): rows.append({"experiment":exp,"status":"missing"}); continue
    d=json.loads(p.read_text()); o=d.get("oof_global",{}); t=d.get("training",{})
    rows.append({"experiment":exp,"status":"complete","folds":t.get("folds"),
                 "best_metric_mean":t.get("best_metric_mean"),"best_metric_std":t.get("best_metric_std"),
                 "oof_tile_rmse":o.get("tile_rmse"),"oof_rmse":o.get("rmse"),"bias":o.get("bias")})
with (out/"comparison.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r})); w.writeheader(); w.writerows(rows)
(out/"analysis_summary.json").write_text(json.dumps({"experiments":rows},indent=2))
print(json.dumps(rows,indent=2))
