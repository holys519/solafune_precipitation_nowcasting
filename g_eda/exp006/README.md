# g_eda/exp006: Exact Mean–Shape and Metric Audit

既存OOF fieldだけで、次のP0診断を再現可能に実行します。

1. tile mean biasとzero-mean spatial residualの厳密な加法直交分解
2. `mean × normalized shape`のmean / shape / scale-shape interaction分解
3. true tile meanによるscaleと、非負L2最適scaleの比較
4. exp016/017/018のfinal predictionから抽出したtile-mean amount component ×
   normalized-field shape componentの3×3 cross-swap
5. tile平均RMSEとglobal pooled RMSEのscale stress
6. own-row observation数別の誤差監査
7. 衛星内層別location-cluster bootstrapによるconditional target-oracle
   opportunityの不確実性評価

## 重要な規約区分

- exp016/017/018 cacheはsuccessor row由来なので、結果は**amber screening限定**です。
- strict finalの判断には、同じ診断をCSV-row-only OOFで再現する必要があります。
- exp011の既存OOF CSVをstrictな加法分解referenceとして併記します。
- true meanやL2最適scaleはtarget oracleです。診断以外に使わず、evaluation予測へ適用しません。
- mean-bias removalも直交分解上の診断です。補正後pixelは負になり得るため、そのまま提出用後処理にはしません。
- multiplicative分解はtarget meanとprediction meanがともに正のtileだけで集計します。
- cross-swapは、各モデルの実装上のhead出力を交換する実験ではありません。final predictionの
  tile meanをamount component、tile meanで1に正規化したfieldをshape componentとして
  再構成する診断です。
- location-cluster bootstrapが評価するのは、target oracleを条件としたimprovement opportunityです。
  学習可能なcalibratorや新モデルのdeployableな効果のCIではありません。
- observation数はtrain CSVの当該rowに列挙されたown-row observationsの数です。successor由来の
  exp016/017/018が実際に入力したframe数を表すものではなく、利用可能性のproxyに限定します。
- evaluation画像、placeholder target、外部データ・外部定数は読みません。

## Inputs

- `outputs/g_eda/exp003/{exp016,exp017,exp018}_oof_pred.npz`
- `outputs/g_eda/exp003/recommended_weights.json`（任意。欠落時はmanifest記録付き既定値）
- `outputs/analysis/exp011/oof_sample_metrics.csv`
- `data/train_dataset/train_dataset.csv`（metadataとrow内観測数だけ）

全予測指標は同一NPZから再計算します。旧`oof_sample_metrics.csv`はexp011 strict reference以外の
予測値joinには使いません。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_eda/exp006
sbatch singularity_run.sh
```

ローカル検証:

```bash
python3 run_factorization_audit.py --self-test
python3 run_factorization_audit.py \
  --max-tiles 512 \
  --bootstrap-reps 100 \
  --skip-tile-csv \
  --out-dir /tmp/g_eda_exp006_smoke
```

## Outputs

`outputs/g_eda/exp006/`へ次を出します。

| File | 内容 |
| --- | --- |
| `EDA_REPORT.md` | 結論、risk、decision rule |
| `factorization_summary.csv` | model × fold/satellite/location/amount等の分解 |
| `tile_factorization.csv` | tile単位の分解・scale oracle診断値（十分統計ではない） |
| `cross_swap.csv` | final-prediction tile-mean amount component × normalized-field shape component 3×3 |
| `cross_swap_lofo.csv` | leave-one-fold-out内側選択とheld-out fold評価 |
| `metric_scale_stress.csv` | scaleごとのtile/global両指標 |
| `tail_concentration.csv` | worst 5/10% tileのnorm/SSE寄与 |
| `strict_exp011_additive.csv` | strict-row-only reference |
| `availability_summary.csv` | own-row observation数別の誤差（successor modelの実入力frame数ではない） |
| `location_bootstrap.csv` | conditional target-oracle opportunityのpaired location-cluster CI |
| `summary.json` | machine-readable主要結果 |
| `run_manifest.json` | 入力path/size/mtime/SHA-256、cache schema・ID contract、引数、script hash |

## Go / No-Go

- true-mean scaleがL2 scale oracle改善の75%以上を4/5 foldsで回収し、strict OOFでも再現した場合、
  `mean × normalized shape`学習A/Bへ進みます。
- cross-swapの採択判断に、全OOFを見て選んだbest pairのpost-selection fold winsは
  使いません。leave-one-fold-outの内側4 foldsでpairを選びheld-out foldで評価した後、
  strict outer-location OOFで固定pairを確認します。`-0.003`はout-of-selection評価にだけ
  適用します。
- hidden metric確定まではtile/global championを別registryで保持します。
