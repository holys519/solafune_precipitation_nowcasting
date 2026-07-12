# exp026: Overlap Patch on exp024 Blend (G-022 followup)

純粋な後処理。LBベスト `exp024_equal_016_017.zip`(0.6919274860606568)には exp014 の
タイル重複GPMコピーパッチが**適用されていなかった**ため、それを適用するだけの実験。
exp009ベースでのパッチ効果は −0.0185(0.71534 → 0.69687)だったので、
期待値は 0.673 前後(= flat tile-mean oracle 0.677 の壁越え)。

## Method

`../exp014/apply_overlap.py` をそのまま呼ぶ(コード複製なし)。config だけ本ディレクトリの
ものを渡す。**config内の相対パスは exp014/ 基準で解決される**点に注意
(`apply_overlap.py` の `resolve_path` が SCRIPT_DIR 基準のため)。

- source: `outputs/submissions/exp024/equal_016_017/`(blend.py が生成済みのディレクトリ)
- output: `outputs/submissions/exp026/` + `outputs/submissions/exp026_submission.zip`

## Run

```bash
cd g_experiments/exp026
sbatch singularity_run.sh
# またはCPUがあるログインノード/ローカルで直接:
#   python3 ../exp014/apply_overlap.py --config config.yaml
```

数秒〜数十秒で完了(GPU不使用)。ログは `outputs/analysis/exp026/run.log`。

## Acceptance

- `patched=` 行が3地点(north_sumatra / northeast_malaysia / sylhet)の行数分カウントされること。
- 提出して 0.6919 からの改善幅を記録(`doc/public_scores.md`)。exp009 のときの −0.0185 と
  同程度なら、パッチ効果がベースに依存しない(=加法的)ことの確認にもなる。
