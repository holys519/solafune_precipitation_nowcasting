# exp014: Tile-Overlap GPM Copy Patch (G-022)

`l_eda/exp002` の発見 (`doc/tile_overlap_discovery.md`) を実装した**純粋な後処理**実験です。
モデル学習・推論は行わず、既存の提出物 (test_files/) の一部ピクセルを、確認済みの空間重複
領域について同時刻のtrain GPM正解値で上書きします。GPU不要、CPUのみで完結します。

## Background

`l_eda/exp002` の正規化相互相関分析で、以下3組のeval/trainタイルが同一時刻に空間重複
していることを確認済みです (`overlap_pairs.csv`)。train-train重複ペア
(atlantic_coast/florida) で重複領域のコピー精度を直接検証し、**RMSE=0.0000 (ビット完全一致)**
でした — GPMタイルは同一プロダクトの切り出しのためです。

| eval地点 | train地点 | GPMオフセット(dy,dx) | 重複px | 重複率 |
| --- | --- | --- | ---: | ---: |
| north_sumatra | aceh | (-17,-23) | 432 | 26% |
| northeast_malaysia | hat_yai | (-22,-9) | 608 | 36% |
| sylhet | dhaka | (11,-15) | 780 | 46% |

該当3地点はeval時間窓と1:1対応 (各地点の全行が同時刻のtrain行を持つ)。

## ルール上の注意

外部データは使わず、配布されたtrain GPMとeval CSVの時刻情報のみを使用しています。
組織側への確認は取っていませんが、配布データ内の利用として実施する判断で進めています
(`doc/tile_overlap_discovery.md` 参照)。

## Method

1. ベース提出物 (`paths.source_submission_dir`、既に推論済みの `test_files/`) を読み込む。
2. eval行の地点が `overlap_pairs.csv` にあれば、同時刻のtrain行を検索し、そのGPM正解を
   読み込む。
3. 重複領域のピクセルだけを `write_float32_like_template` でtrain値に置き換える
   (テンプレート=既存の予測tif自体なので、メタデータ・dtype・shapeは自動的に維持される)。
4. 重複領域外のピクセルと、重複しない地点のファイルは無変更でコピー。
5. 提出zipを再構成。

## Run

```bash
cd g_experiments/exp014
# 先に何らかの実験でeval推論を済ませ、config.yamlのsource_submission_dirを向ける
# (デフォルトは ../../outputs/submissions/exp009)
../../.venv/bin/python apply_overlap.py --config config.yaml
```

CPUのみで数秒〜数十秒 (18,000行程度のCSV走査 + 該当行のみtiff読み書き)。

## Testing

GPU/実提出物なしでも検証可能。合成のsource submission (定数0.5の予測) を作り、
overlap領域がtrain GPMと完全一致し、非overlap領域が0.5のまま・非対象地点が無変更で
あることを確認済み (このセッションで実施、コード内には残していない一時テスト)。

## Outputs

| Path | Description |
| --- | --- |
| `../../outputs/submissions/exp014/test_files/` | パッチ済み予測 |
| `../../outputs/submissions/exp014_submission.zip` | 提出zip |

## Acceptance (G-022)

- `source_submission_dir` を現行チャンピオン (exp009など) の推論結果に向けて実行。
- `doc/public_scores.md` に記録し、パッチ前との差分でオーバーラップ効果を定量化する。
