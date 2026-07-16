# g_eda/exp005: IMERG生成物理の深掘り (H1/H2/H3/E-9)

`doc/imerg_physics_notes.md` の仮説検証。目的変数 (GPM IMERG) が「PMW通過 + 移流モーフ
ィング + IRフィル」の合成プロダクトであることと、静止衛星の視差幾何から、まだ使って
いない法則性を探す。

| Script | 仮説 | 出力 |
| --- | --- | --- |
| `run_imerg_innovation.py` | H1 (フレッシュ/モーフ二状態)、H2 (移流persistence)、E-9 (日周期) | `innovation_pairs.csv`, `innovation_hist.csv`, `diurnal_cycle.csv`, `IMERG_INNOVATION.md` |
| `run_parallax_geometry.py` | H3 (視差の幾何予測とeval外挿) | `eval_parallax_prior.csv`, `parallax_fits.json`, `PARALLAX_GEOMETRY.md` |

`locations.py` の座標は**分析専用の近似値**。提出パイプラインに座標由来特徴を入れる
場合はGeoNames/Nominatimで再導出し提出説明に記載する
(`discussion/geocoding_coordinates_ja.md` のルーリング参照)。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_eda/exp005
sbatch singularity_run.sh
```

## 使い道 (仮説が通った場合)

- H1 → フレーム状態に応じた適応時間平滑化 / PMWフレッシュ限定の診断指標
- H2 → 予測の「移流付き平滑化」(現在の静的平滑化 −0.0045 の上位互換候補)
- H3 → eval地点への入力レジストレーション補正 (G-017のeval側解禁)
- E-9 → local solar hour sin/cos 入力 (Round 6、閉形式で許可済み)
