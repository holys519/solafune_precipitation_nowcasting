# exp027: Seed-Ensemble Blend + Overlap Patch

exp025で学習済みの exp017×seed{42,123,2026} チェックポイント(`g_model/exp025/seed{s}/`)を
活用するハーベスト実験。学習なし(推論のみGPU使用)。

背景: exp025の実測でfold単位のシード分散が非常に大きい(fold1: 0.469〜0.731、
fold3: 0.816〜1.085)ため、シード平均による分散低減の期待値が大きい。
LBでは exp016+exp017 の均等ブレンドが最良(0.6919)で、exp009 は混ぜるほど悪化。

## Method(run.py、1ジョブで全ステップ)

1. **シード推論**: 各seedについて `outputs/analysis/exp025/seed{s}/config.yaml` を
   `paths.output_dir` だけ上書きして `exp017/inference.py` を実行(5-foldアンサンブル)。
   出力: `outputs/submissions/exp027/seed{s}/test_files/`。既に予測があればスキップ
   (`--force-inference` で再実行)。
2. **ブレンド**(いずれも overlap パッチ前の生ブレンド):
   | scheme | 重み |
   | --- | --- |
   | `half016_half017family` | exp016 = 0.5、exp017本体+3シードで残り0.5を等分(各0.125)— LBで勝った50/50バランスを維持 |
   | `equal_all` | 5ソース均等(各0.2)— exp017ファミリー寄り |
3. **パッチ**: 各schemeに `exp014/apply_overlap.py` を適用(configは絶対パスで自動生成)。

## Outputs

- `outputs/submissions/exp027_half016_half017family_patched.zip` ← 本命
- `outputs/submissions/exp027_equal_all_patched.zip`
- パッチ前ディレクトリも `outputs/submissions/exp027/<scheme>/` に残る(重み再探索用)
- manifest: `outputs/analysis/exp027/analysis_summary.json`

## Run

```bash
cd g_experiments/exp027
sbatch singularity_run.sh              # seeds 42,123,2026
sbatch singularity_run.sh 42,123       # サブセット
```

シード推論 ~30-60分/seed、ブレンド+パッチは数分。

## Acceptance

- 本命zipのLBが exp026(パッチ済み016+017ブレンド)を上回ること。上回らなければ
  「シード追加はLBでは飽和」の証拠として記録し、以後はexp018(局在化)に集中する。
