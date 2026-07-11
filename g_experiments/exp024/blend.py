#!/usr/bin/env python3
import json, shutil, zipfile
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp017"))
from tiff_utils import read_tiff_array, write_float32_like_template
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/"outputs"; ANA=OUT/"analysis/exp024"; ANA.mkdir(parents=True,exist_ok=True)
sources={e:OUT/"submissions"/e/"test_files" for e in ("exp009","exp016","exp017")}
schemes={"equal_009_016_017":{"exp009":1/3,"exp016":1/3,"exp017":1/3},"equal_016_017":{"exp016":.5,"exp017":.5},"blend_20_40_40":{"exp009":.2,"exp016":.4,"exp017":.4}}
manifest=[]
for name,weights in schemes.items():
 files={e:{p.name:p for p in sources[e].glob("*.tif")} for e in weights}; common=sorted(set.intersection(*(set(x) for x in files.values())))
 if not common: manifest.append({"name":name,"status":"missing_predictions","weights":weights}); continue
 dest=OUT/"submissions/exp024"/name/"test_files"; dest.mkdir(parents=True,exist_ok=True)
 for fn in common:
  ref=files[next(iter(weights))][fn]
  arr=sum(w*read_tiff_array(files[e][fn])[0].astype(np.float32) for e,w in weights.items())
  write_float32_like_template(ref,dest/fn,arr)
 csv_src=ROOT/"data/evaluation_dataset/evaluation_target.csv"; shutil.copy2(csv_src,dest.parent/"evaluation_target.csv")
 zpath=OUT/"submissions"/f"exp024_{name}.zip"
 with zipfile.ZipFile(zpath,"w",zipfile.ZIP_DEFLATED) as z:
  z.write(dest.parent/"evaluation_target.csv","evaluation_target.csv")
  for p in sorted(dest.glob("*.tif")): z.write(p,f"test_files/{p.name}")
 manifest.append({"name":name,"status":"complete","weights":weights,"files":len(common),"zip":str(zpath)})
(ANA/"analysis_summary.json").write_text(json.dumps({"schemes":manifest},indent=2)); print(json.dumps(manifest,indent=2))
