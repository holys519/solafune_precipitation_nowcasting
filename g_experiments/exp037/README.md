# exp037: 8-view TTA 再推論 + exp036 combo

現状のTTAはflip 3-viewのみで、学習augmentationに含まれるrot90群が推論側で未使用だった
(公開0.69解法は6-way、G-034でも計画済みのまま未実施)。exp016/017/018のinference.pyに
`tta.rot90` を追加し (デフォルトfalse、既存実験の挙動不変)、8-view (identity + h/v flip +
rot90/180/270) で eval を再推論 → exp036のcombo (衛星別重み + blur σ1.0 + threshold 0.2)
→ overlap patch。

対照は `exp036_per_satellite_blur1_thr0p2_patched.zip` (Public 0.6706858062196032)。
差分は**rot90 TTAのみ**なので、これが8-view TTAの実測A/Bになる。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp037
sbatch singularity_run.sh                 # 3モデル再推論 → blend → patch → zip
sbatch singularity_run.sh --blend-only    # 推論済み予測からblend以降のみ
```

## Outputs

- 再推論予測: `outputs/submissions/exp037/{exp016,exp017,exp018}/test_files/`
- 提出zip: `outputs/submissions/exp037_per_satellite_blur1_thr0p2_patched.zip` (+ raw zip)
- manifest: `outputs/analysis/exp037/analysis_summary_per_satellite_blur1_thr0p2.json`

## 期待値と判断

- ディスカッション実測ではTTA/seed系は各+数千分の一。E-3ノイズ帯 (~0.004) 未満の
  可能性が高いが、**同一提出物に他の改善と同乗させる恒久基盤**になる (以後のblendは
  すべてexp037のTTA済み予測をsourceにする)。
- 提出枠に余裕がある日に1枠で検証。悪化しなければsourceを恒久切替。
