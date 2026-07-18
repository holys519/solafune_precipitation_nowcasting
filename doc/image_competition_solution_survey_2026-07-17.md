# 画像コンペ・降水コンペから抽出する安全な解法パターン

- 最終更新: 2026-07-17
- 対象: Solafune 静止気象衛星画像からの GPM-IMERG 降水量推定
- 位置づけ: 外部コンペの手法を、配布データだけで検証できる仮説へ翻訳するための method-transfer inventory
- 関連文書: [研究調査 Round 3](research_survey_v3_2026-07-16.md)、[コンペ規約](competition_rules.md)、[実験計画](../g_experiments/EXPERIMENT_PLAN.md)、[exact factorization EDA](../outputs/g_eda/exp006/EDA_REPORT.md)

> [!IMPORTANT]
> 本稿は外部データを導入する提案ではない。公開コンペや論文から **algorithmic pattern と検証方法だけ** を抽出し、Solafune 配布データのみを使う controlled A/B へ翻訳する。外部コンペでの成功は、本課題での有効性を保証しない。

## 0. 結論

現在の最優先は、モデルをさらに大きくすることではなく、予測を次の三要素へ分解して検証することである。

1. **Amount**: その tile にどれだけ降るか
2. **Shape**: tile 内のどこに、どの形で降るか
3. **Lifecycle / domain state**: 雲が発達中・成熟・衰弱のどこにあり、どの衛星・観測条件で見えているか

第一候補は、exp035 `no_dilation` の **architecture だけ** を参考にし、strict current-row-only で scratch から再学習する **tile amount × normalized shape** である。既存 `config_no_dilation.yaml` / checkpoint は `context_rows: 2` の successor context と外部仕様由来の fixed band mapping / coefficients を含むため Amber であり、strict base にはしない。`g_eda/exp006` では Amber の exp018 OOF tile RMSE が `0.609271` であるのに対し、真の tile mean を使う診断的 scale oracle は `0.544628`、L2 scale oracle は `0.535498` で、scale oracle 改善の `87.6%` を真の mean だけで回収した。これは amount 仮説の強い screening evidence だが、strict OOF で再現して初めて Green evidence になる。

第二候補は、**motion-aligned cloud lifecycle hysteresis** である。同じ最新時刻の雲頂状態でも、発達中と衰弱中では IMERG target が異なる、という仮説を、同一行の過去フレームだけで検証する。絶対的な温度閾値は使わず、fold 内で学習した DN rank、texture、雲域面積、整列後 trend を用いる。

第三候補は、hard な衛星別出力 head ではなく、shared model に対する **soft sensor conditioning** である。`exp032` の satellite-conditional output heads は fold0 で悪化したが、feature-level FiLM は情報共有を残すため別仮説である。

最小の優先順は次である。

1. `XC-01` amount × normalized shape
2. `XC-02` lifecycle hysteresis EDA
3. `XC-03` lifecycle-conditioned amount head
4. `XC-04` feature-level sensor FiLM
5. `XC-05` shared temporal encoder + feature differences
6. `XC-06` nested rain-threshold auxiliary heads

`XC-07` 以降は、上位案の結果を見てから着手する。全案は 2-fold screening だけで採択せず、最終的に strict 5-fold OOF と location-cluster bootstrap で判断する。

主要な転用関係は次のとおりである。

| Source pattern | 本課題での転用 | Proposal | Priority | 主な誤用リスク |
| --- | --- | --- | --- | --- |
| exp006 amount oracle + global scalar heads | amount × normalized shape | XC-01 | P0 | target oracle を推論に使う |
| convective lifecycle / semi-Lagrangian tracking | aligned temporal trend | XC-02/03 | P0/P1 | uint8 を Kelvin や冷却率と断定する |
| Weather4Cast region conditioning | shared feature FiLM | XC-04 | P1 | location shortcut、hard domain split |
| xView2 shared temporal fusion | per-frame encoder + feature differences | XC-05 | P1 | cross-row frame の混入 |
| SpaceNet/Data Science Bowl auxiliary masks | nested rain contours | XC-06 | P1 | IoU 改善を continuous RMSE 改善とみなす |
| Open Cities rare-tile sampling | corrected regime sampler | XC-07 | P2 | calibration を壊す過剰 sampling |
| SaTformer distributional scalar target | binned tile amount posterior | XC-08 | P2 | CRPS の成功を RMSE へ直結させる |
| HighRes-net registered loss | shape-only small-shift auxiliary | XC-09 | P2 | exact pixel loss を置き換える |
| SpaceNet 7 high-resolution branch | HRNet-lite shape decoder | XC-10 | P2 | 計算量だけ増やす |
| radar sequence drop-in | age-aware frame drop/repeat | XC-11 | P3 | 実際の欠損分布から逸脱する |
| Weather4Cast band permutation | conditional band diagnostics/dropout | XC-12 | P2/P3 | 他 sensor の band 解釈を固定移植する |

## 1. 読み方と証拠階層

この文書では、先行研究の事実と本課題への推論を混同しない。

| Label | 意味 | 記述例 |
| --- | --- | --- |
| Source claim | 論文・主催者・著者が、その課題について報告したこと | 「補助 head を最終 pipeline で使用した」 |
| Local evidence | この repository の OOF、EDA、public score が示すこと | 「true-mean scaling に大きい診断的 headroom がある」 |
| Transfer inference | source claim と local evidence から立てる本課題固有の仮説 | 「tile amount head を独立に学習すると改善し得る」 |
| Confirmed result | strict 5-fold OOF と事前定義 gate を通ったこと | 実験完了後だけ使用する |

「winner が使用した」は「その要素単独の改善が ablation で証明された」と同義ではない。特に主催者 blog の最終 pipeline、ensemble、pseudo-labeling は交絡が大きいため、本稿では `reported/used` と `improved in an isolated ablation` を分ける。

## 2. 現在のローカル証拠

### 2.1 Amount が最大の未回収候補

`outputs/g_eda/exp006/EDA_REPORT.md` の exact decomposition は、additive mean bias と multiplicative amount error を区別している。ただし exp016/017/018 は successor-context を含む Amber artifacts なので、この節は architecture / objective を選ぶ screening であり、最終解法の Green evidence ではない。

| exp018 diagnostic | Tile RMSE | Actual からの改善 | 解釈 |
| --- | ---: | ---: | --- |
| actual OOF | 0.609271 | - | 現行予測 |
| mean-bias removed | 0.586905 | -0.022366 | additive bias の寄与 |
| true-mean scale oracle | 0.544628 | -0.064643 | tile amount を正しくした場合の診断値 |
| L2 scale oracle | 0.535498 | -0.073773 | 各 target を見た最大 scale 診断値 |

- true-mean scale は L2 scale oracle gain の `87.6%` を回収する。
- additive mean SSE share は `8.1%` であり、単純な定数 bias 補正だけでは不足する。
- exp016/017/018 の amount/shape cross-swap は tile 指標の leave-one-fold-out gate を `0/5` しか通らない。したがって、既存モデル間で部品を交換するより、factorization を training objective にする方が筋がよい。
- これらの target oracle は **診断専用** であり、evaluation 推論や submission post-processing には使用しない。
- amount headroom の大きさと回収率は、同じ分解を strict current-row-only OOF で再計算して確認する。

### 2.2 Amber architecture screening と strict baseline の要件

2026-07-17 時点の `g_experiments/EXPERIMENT_PLAN.md` では、exp035 の `no_dilation` が fold0 で `0.28694`、fold4 で `0.58538` であり、exp018 の `0.29234 / 0.58531` に対して fold0 は改善、fold4 は同等である。fold1--3 は実行中であるため採択結果ではなく、さらに両者とも Amber input pipeline 上の architecture screening である。

現行 `g_experiments/exp035/config_no_dilation.yaml` は次を含む。

- `context_rows: 2` と successor row の newest frame
- 外部仕様に基づく satellite 別 canonical band index
- fixed engineered ratios / differences / thresholds

Green の strict baseline は、`context_rows=1`、own-row observations のみ、外部仕様 mapping / fixed physical coefficients なし、既存 Amber checkpoint を load せず scratch training、という新 config で作る。exp035 の残り folds は architecture 選択の情報にはなるが、完了しても strict base の確定にはならない。

- `dilation_only` は fold0 と fold4 の両方で悪化し、優先度を下げる。
- `tilemean` auxiliary は fold0 `-0.0025`、fold4 `+0.0007` と不安定である。
- したがって、scalar auxiliary を足すだけでなく、出力を amount と normalized shape に構造的に分ける A/B が必要である。
- advected prediction smoothing は `+0.00012` で棄却されている。motion は出力平滑化より、入力時系列の整列と lifecycle feature に使う。

### 2.3 評価指標と検証単位

ローカルの submitted-pair 解析では、per-file-averaged RMSE に対応する OOF tile RMSE が leaderboard と高い順位相関を示す。最新記録は tile Spearman `0.961`、pooled-pixel `0.681` である。ただし運営の評価実装が書面で確認されるまでは、これは **強い経験的同定** と表現し、公式定義確定とは言い切らない。

- primary: per-file RMSE の平均に対応する tile RMSE
- secondary: 全 pixel pooled RMSE
- acceptance noise floor: OOF-to-LB 校正から約 `0.004--0.005`
- fold0 単独は弱い。fold0+4 は screening のみとする。
- location が独立単位なので、主な uncertainty interval は row bootstrap ではなく location-cluster bootstrap を使う。

### 2.4 時系列可用性

own-row observation count は `0: 235`, `1: 8`, `2: 647`, `3: 39796` である。ほぼ全行が 3 観測を持つため、欠損頑健化は必要だが、現時点では amount factorization より優先しない。また successor-row context を使った既存 artifact と strict current-row-only artifact を混ぜない。

## 3. 基礎物理から見た問題設定

### 3.1 これは単純な画像変換ではなく、非一意な逆問題である

静止気象衛星が主に観測するのは雲頂・水蒸気・放射特性であり、target は地表付近の降水 product である。似た雲頂放射でも、雲の相、鉛直発達、組織化、地表へ落下するまでの移流、target product の retrieval state により降水量は変わる。

したがって、1 pixel ごとの `IR value -> rain` の固定写像だけでは足りない可能性が高い。次はモデル化候補となる潜在要因であり、すべてを明示状態として持つ必要が証明されたわけではない。

- instantaneous cloud state: 最新フレームの冷雲域、texture、multi-band contrast
- lifecycle state: 発達・成熟・衰弱
- motion state: 雲系の移動方向と整列 confidence
- sensor/domain state: 衛星、band availability、欠損、観測条件
- target-product state: IMERG が持つ retrieval / interpolation の混合性

JMA の CCI/RDCA 解説や deep convective cloud tracking 研究では、multi-band differences と時間変化を対流雲の発達評価に使う。ただし本データは `uint8` で calibration provenance が確定していないため、Kelvin の絶対閾値を移植しない。[JMA CCI/RDCA](https://www.data.jma.go.jp/mscweb/technotes/msctechrep62-2.pdf)、[semi-Lagrangian DCC tracking](https://amt.copernicus.org/articles/16/1043/2023/)

### 3.2 Amount × normalized shape

予測を次のように構成する。

```text
a      = softplus(amount_mlp(global_features))
s0     = softplus(shape_logits)
s      = s0 / clamp_min(mean_pixel(s0), eps)
y_hat  = a * s
```

この定義では `mean_pixel(s) ≈ 1` なので、`mean_pixel(y_hat) ≈ a` となる。amount head と shape decoder が同じ尺度を奪い合う scale ambiguity を減らし、global context と local texture に役割を分けられる。

重要な注意は次である。

- dry tile では normalized target shape が未定義または不安定なので、shape auxiliary を skip する。
- main loss は必ず最終 `y_hat` に対して計算し、factorized component の loss だけで代用しない。
- wet-mask thresholding を推論へ入れるなら、その後 shape を再正規化しないと amount が変わる。
- amount target、dry threshold、log transform の定数は training fold だけから決める。
- true target mean を使うのは training supervision と診断だけであり、evaluation 推論では amount head の出力を使う。

### 3.3 Lifecycle hysteresis

物理仮説は「最新時刻の雲頂放射状態が同じでも、雲が発達中か衰弱中かで target rain distribution が異なる」である。強い上昇流に伴う雲頂上昇、雲域拡大、roughness や氷雲特性の変化に対し、成熟後は地表降水が弱まっても冷たい anvil が残り得る。そのため、同じ最新状態に成長期・成熟期・衰弱期が混在し得る。

ただし現在の入力は `uint8` であり、DN の単調方向、band identity、放射 calibration は未確認である。最初の EDA では物理的な `cooling / growth / decay` と命名せず、経験的な trend quantile を **`D- / stable / D+`** と呼ぶ。calibration と band identity を確認できた場合も、まず言えるのは `cloud-top cooling/warming proxy` までである。`convective growth/decay` と呼ぶには、object area、roughness、複数 band、追跡継続性との整合、可能なら独立観測を追加で要求する。

raw difference は motion と lifecycle を混同する。まず過去フレームを最新フレームへ小さい探索範囲で整列し、その後に trend を測る。

```text
past frame -- small-shift alignment --> aligned past
latest - aligned past              --> lifecycle trend
NCC gain / displacement            --> confidence and motion state
```

局所 box 平均だけでは強い convective core の変化を anvil で希釈し得ることが対流追跡研究で報告されているため、tile mean だけでなく core quantile、empirical cloud-proxy area、texture change を残す。[SATCAST object tracking](https://journals.ametsoc.org/view/journals/apme/51/11/jamc-d-11-0246.1.xml)、[PERSIANN-CCS cloud-patch classification](https://journals.ametsoc.org/view/journals/apme/43/12/jam2173.1.xml)

この仮説は、次の二つの終了済み仮説とは別である。

- prediction field 自体を移流して平滑化する案は `+0.00012` で棄却済みである。
- target source を `PMW fresh / morphed` の二状態とみなす案は、ローカル EDA で支持する二峰性がなく棄却済みである。

ここで検証するのは、**同一行の衛星 3 frame から material-like temporal derivative を作り、latest state にない追加説明力を調べること** である。

### 3.4 Target は独立した雨量真値ではなく IMERG product である

一般的な IMERG V07 は PMW retrieval、PMW 観測間の CMORPH-KF 系 quasi-Lagrangian interpolation、PDIR 系 IR retrieval などを品質情報とともに統合する衛星降水 product である。一般的な V07 のアルゴリズム知識は、target が持ち得る滑らかさ、source-dependent uncertainty、履歴依存を理解するために使える。[NASA IMERG V07 documentation](https://gpm.nasa.gov/resources/documents/imerg-v07-technical-documentation)、[IMERG V07 ATBD](https://gpm.nasa.gov/sites/default/files/2023-07/IMERG_V07_ATBD_final_230712.pdf)、[CMORPH](https://www.ftp.cpc.ncep.noaa.gov/precip/CMORPH_V1.0/REF/Joyce_et_al_2004_JHM_CMORPHP.pdf)

ただし、本コンペ target がどの run、version、処理時刻、入力 availability で生成されたかはローカル情報だけでは確定しない。したがって、target が Early/Late/Final のどれか、個々の pixel が PMW/IR/morph のどの source か、時間平滑化の応答が IMERG morphing に由来するか、future-side 情報の効き方が双方向 morphing を証明するか、を断定しない。

PDIR では cloud-patch temperature、size、texture、cloud type などを用いる。これは patch/object-level conditional retrieval の先例であり、lifecycle conditioning の直接的先例は JMA CCI/RDCA と semi-Lagrangian tracking である。PDIR の閾値や外部 calibration を直接利用してはならない。[PDIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC8216223/)

コンペ内で主張できるのは **配布 satellite image から IMERG target を再構成する性能** である。実雨量の改善、物理的因果、DPR に対する精度向上を主張するには、コンペ終了後に完全分離した外部検証が必要である。

## 4. 降水・気象コンペからの転用候補

### 4.1 Weather4Cast 2025: distributional amount prediction

- **Source claim**: SaTformer は 1 時間・4 時刻・11 bands の衛星入力から、32×32 radar 領域の平均 4-hour cumulative rainfall の eCDF を予測する設定で、64-bin classification、inverse-frequency weighted categorical CE、full space-time attention を用いた。これは 32×32 の降水画像を生成する課題ではない。2026-07-17 閲覧時の公式 Cumulative Rainfall Task #1 leaderboard では首位だが、論文中の validation BW-CRPS と公式 leaderboard score は別物である。[SaTformer paper](https://arxiv.org/abs/2511.11090)、[official task](https://weather4cast.net/neurips2025/competitions/w4c25-cum1/)、[official leaderboard](https://weather4cast.net/neurips2025/competitions/w4c25-cum1/?leaderboard=)
- **Transfer inference**: 本課題で直接対応するのは dense field 全体ではなく tile amount distribution である。amount を 32/64 bins の distribution として予測し、期待値または校正済み代表値を factorization の `a` に使える。
- **Caveat**: frequency weighting は rare heavy rain を学びやすくする一方、確率 calibration を歪め得る。unweighted、tempered weighting、continuous deterministic head を同一 OOF で比較する。
- **Priority**: P2。既存の quantile serving が負であり、まず deterministic factorization を試す。
- **License**: Hugging Face checkpoint は Apache-2.0 表示を確認できるが、GitHub code root には明示的な LICENSE を確認できない。さらに model card は base model に Kinetics-400 由来の TimeSformer を示すため、checkpoint license、code license、base-model license、training-data provenance を別々に監査する。現コンペで checkpoint を使う提案ではなく、concept-only clean-room 実装とする。[checkpoint card](https://huggingface.co/leharris3/satformer)、[code repository](https://github.com/leharris3/satformer)

### 4.2 Weather4Cast 2023: controller、interpolation、sampling

Weather4Cast 2023 の各 track は課題設定が異なるため、総合順位としてまとめない。[official challenge page](https://weather4cast.net/neurips2023/)

1. **Lightweight heads + controller**
   - Source claim: 複数の lightweight prediction heads と controller、段階的学習を用い、高降水域への対応を報告している。[solution paper](https://arxiv.org/abs/2401.09424)
   - Transfer inference: shared shape trunk に複数の soft amount experts を置き、rain regime に応じて mixture する。
   - Stop rule: hard expert assignment、衛星別 output head の再実装にはしない。shared deterministic amount head を超えなければ終了する。

2. **Temporal interpolation + multi-level Dice**
   - Source claim: 時間 frame interpolation と複数閾値に対する Dice 系 objective を使用した。[solution paper](https://arxiv.org/abs/2311.18341)
   - Transfer inference: own-row の欠損補間と nested rain-threshold auxiliary heads に分けて検証する。
   - Caveat: target-time future frame を他行から取得することは別問題であり、strict green 実験では行わない。

3. **Importance sampling / context crop / probabilistic output**
   - Source claim: RainAI は dataset preparation、sampling、classification/probabilistic formulation、context crop を含む pipeline を報告した。[solution paper](https://arxiv.org/abs/2311.18398)
   - Transfer inference: satellite × amount bin × wet fraction の stratified sampler と、tile amount classification を別々に A/B する。
   - Caveat: sampler の効果と probabilistic head の効果を同時に変更しない。

### 4.3 Weather4Cast 2022: soft domain conditioning と band diagnostics

Weather4Cast 2022 は satellite frames から future rain-event masks を約 6 倍に super-resolve し、IoU 系指標で評価する課題である。本課題の continuous IMERG regression / current-time retrieval と同一ではないため、順位や IoU improvement を RMSE gain に換算しない。[official challenge report](https://proceedings.mlr.press/v220/gruca23a.html)

1. **Region-conditioned 3D U-Net**
   - Source claim: region-dependent layers / conditioning、self-distillation、mixup、orthogonality regularization を含む solution が、著者評価で cross-region transfer 改善を報告した。[paper](https://arxiv.org/abs/2212.02059)
   - Transfer inference: satellite ID と availability metadata から feature-level FiLM の scale/bias を作る。出力 head は共有する。
   - Local evidence: `exp032` の hard satellite-specific heads は fold0 で悪化した。これは hard separation を閉じる証拠であり、soft FiLM を直接否定しない。
   - Rule constraint: train location ID を conditioning に使わない。未見 location で shortcut になるためである。

2. **3D U-Net / EarthFormer and band permutation**
   - Source claim: context length、loss ensemble、band permutation importance を調べ、同課題の permutation test では `8.7 / 10.8 / 13.4 µm` bands の重要性を報告した。[paper](https://arxiv.org/abs/2212.02998)
   - Transfer inference: 本データの各 satellite で、conditional permutation と band-group dropout を行う。
   - Caveat: Weather4Cast の band 番号と本データの band を固定対応させない。AHI/ABI/FCI の spectral response と現在の `uint8` preprocessing が異なるため、重要度は本データ内で再推定する。
   - Code audit: 公開 repository が Apache-2.0 であっても、使用時は commit、license、取得日、SHA-256 を manifest に残す。[repository](https://github.com/bugsuse/weather4cast-2022-stage2)

3. **WeatherFusionNet**
   - Source claim: satellite-to-radar branch、future satellite prediction、raw temporal sequence を融合した。論文の held-out IoU は core で `0.3162` 対 U-Net `0.2950`、transfer で `0.2488` 対 U-Net `0.2567` と報告され、core の改善が transfer へ自動的に移らない。[paper](https://arxiv.org/abs/2211.16824)
   - Transfer inference: auxiliary future-frame prediction は source-domain fit を良くしても unseen-location generalization を悪化させ得る。採用判定は location-held-out のみに置く。
   - License: repository は Apache-2.0 を明記しているが、本稿では data や weights を使用しない。[repository](https://github.com/Datalab-FIT-CTU/weather4cast-2022)

### 4.4 Weather4Cast 2024: optical flow + cGAN

- **Source claim**: IR radiance の optical-flow extrapolation と conditional GAN を組み合わせた solution を、著者は challenge 時点の 1 位と報告した。2026-07-17 閲覧時の live leaderboard では後日の entries を含み順位が異なるため、「現在の 1 位」とは書かない。論文自身も peak underestimation と spatial misalignment を限界として挙げる。[paper](https://arxiv.org/abs/2412.00451)、[live leaderboard](https://weather4cast.net/neurips2024/competitions/w4c24-cum1/?leaderboard=)
- **Transfer inference**: 借りるのは target-time への低自由度 warp と alignment confidence だけである。
- **Do not transfer**: GAN を final regression field の主 objective にしない。RMSE と calibration に対する直接性が低く、現時点の主要残差である amount を解かない。
- **License**: author repository の root LICENSE は未確認なので、code をコピーせず concept-only とする。[repository](https://github.com/flame-cai/Weather4Cast24_NIPS)

### 4.5 Weather4Cast 2021 と他の降水コンペ

- **Shortcut ConvGRU / rain regimes**: conditional rain-regime design は、shared trunk + soft amount experts の先例として扱う。ただし元課題の gate は provided input の `crr_intensity` を利用できた。本課題には同じ weather product がないため、外部 product を導入せず own-row radiance から学ぶ soft proxy とし、hard routing は避ける。[paper](https://arxiv.org/abs/2111.06240)
- **Kaggle How Much Did It Rain II**: winner interview では variable-length radar sequence に対する temporal drop-in / duplication が説明される。本課題では own-row frame の drop/repeat、age channel、valid/duplicate mask に翻訳する。[winner interview](https://medium.com/kaggle-blog/how-much-did-it-rain-ii-winners-interview-1st-place-pupa-aka-aaron-sim-48c9c24473c1)
- **DrivenData Tropical Storm**: shared CNN、recurrent history、time augmentation、weighted ensemble という pattern が報告される。本課題では shared per-frame encoder と global amount head が対応する。[winner summary](https://drivendata.co/blog/wind-dependent-variables-winners)

## 5. 一般画像・地球観測コンペからの転用候補

### 5.1 Open Cities AI Challenge: scalar auxiliary と rare-tile sampling

- **Source claim**: 主催者の winner 解説では、強い augmentation、overlap crop、rare-tile sampling、image-level classification や scale regression などが最終 solutions の構成要素として説明される。[official winner write-up](https://drivendata.co/blog/open-cities-disaster-winners/)
- **Critical caveat**: 3 位 solution の scale regression は入力画像の異なる sample scale を扱う補助 head であり、降水 amount ではない。また単独改善の ablation は示されていない。
- **Transfer inference**: image-level scalar head の競技先例ではあるが、tile amount head の直接根拠は Open Cities ではなく `g_eda/exp006` である。
- **Useful patterns**:
  - overlap crop: tile edge artifact を抑える inference technique
  - rare positive sampling: heavy-rain tile を batch に確保する
  - scalar auxiliary: global encoder が tile-level state を保持するよう補助する
- **Do not transfer**: test/tier-2 pseudo-labeling。別コンペで許可されていても、本競技の evaluation adaptation として green にはしない。

### 5.2 xView2: raw concat ではなく shared temporal feature fusion

- **Source claim**: pre/post imagery を shared-weight encoders で独立に処理し、feature fusion する構成が、raw 6-channel concatenation や単純 subtraction より良いと報告された。[paper](https://arxiv.org/abs/2004.05525)
- **Transfer inference**: 各 own-row frame を shared encoder に通し、latest feature、signed difference、absolute difference、valid mask を decoder へ渡す。
- **Why it fits**: `exp028` で latest-frame centric input と `|Δ|` が fold0 改善を示しており、pixel-level concat から feature-level fusion への次の一手になる。
- **Rule constraint**: 同一 CSV 行に紐づく過去観測だけを使う。隣接 row や successor row から frame を補完しない。

### 5.3 PROBA-V HighRes-net: registration-aware auxiliary

- **Source claim**: HighRes-net は multi-image super-resolution で implicit co-registration、recursive fusion、registered loss を提案した。[paper](https://arxiv.org/abs/2002.06460)、[official challenge](https://kelvins.esa.int/proba-v-super-resolution/)
- **Critical caveat**: registered loss は PROBA-V の LR/HR sensor misregistration を補正するもので、雲頂から地上降水までの parallax/advection と同じ現象ではない。また reference-frame heuristic が ranking に影響するという再評価もある。[PROBA-V-REF](https://arxiv.org/abs/2101.10200)
- **Transfer inference**: exact field MSE を主 loss に残し、normalized shape にだけ `±1` または `±2` pixel の shift-softmin auxiliary を低 weight で加える。
- **Stop rule**: exact tile RMSE を改善せず shift-tolerant diagnostic だけ改善するなら終了する。submission field を事後的に target oracle shift しない。

### 5.4 SpaceNet 7: high-resolution branch と boundary auxiliary

- **Source claim**: SpaceNet 7 の challenge report は multi-temporal building tracking の上位手法を比較し、高解像度 backbone、upsampling、footprint/boundary/contact heads などを記録している。[challenge report](https://arxiv.org/abs/2102.11958)
- **Transfer inference**: HRNet-lite または native 41×41 high-resolution branch、rain footprint / nested threshold / gradient auxiliary を試す。
- **Caveat**: SpaceNet の temporal collapse は evaluation sequence 全体を使える設定に依存する。本課題で cross-row temporal collapse を行うと successor-context risk があるため採用しない。

### 5.5 SpaceNet 4: 物理的に妥当な augmentation だけを使う

- **Source claim**: off-nadir building extraction では viewing direction、boundary/contact targets、loss、augmentation の扱いが議論されている。[organizer analysis](https://medium.com/the-downlinq/a-deep-dive-into-the-spacenet-4-winning-algorithms-8d611a5dfe25)、[official solution repository](https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions)
- **Transfer inference**: 地球観測画像では D4 symmetry が自動的に成立しない。時刻、scan direction、view geometry、移流方向との関係を壊す augmentation は検証が必要である。
- **Local evidence**: rotation/D4 TTA はローカルで負なので、再優先化しない。

### 5.6 NTIRE burst super-resolution: alignment confidence

- **Source claim**: burst super-resolution の上位手法は、複数フレームの alignment と fusion を主要要素とする。[NTIRE 2021 report](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Bhat_NTIRE_2021_Challenge_on_Burst_Super-Resolution_Methods_and_Results_CVPRW_2021_paper.html)
- **Transfer inference**: dense optical flow をそのまま導入せず、`±2 px` 程度の discrete NCC alignment、correlation gain、estimated displacement を lifecycle feature にする。
- **Reason for low freedom**: 41×41 target と短い own-row sequence では、高自由度 flow は雲の生成消滅を motion と誤認しやすい。

### 5.7 Data Science Bowl nuclei: nested regions と boundary learning

- **Source claim**: nuclei segmentation solutions では interior、boundary、relative-distance など複数の構造 target を併用する方法が分析された。[competition report](https://www.nature.com/articles/s41592-019-0612-7)
- **Transfer inference**: rain field について `y > threshold` の nested masks、Sobel gradient、wet-region distance transform を auxiliary targets にする。
- **Use at inference**: auxiliary heads は捨て、final continuous field だけを出力する。threshold mask で hard clipping する案とは分離する。

### 5.8 DSTL、Cloud Cover、Kelp、STAC

- **DSTL satellite imagery**: class-specific bands/scales と rare-positive sampling の pattern を、satellite-specific band group と wet-tile sampling の診断に翻訳する。[winner interview](https://medium.com/kaggle-blog/dstl-satellite-imagery-competition-1st-place-winners-interview-kyle-lee-6571ce640253)
- **Cloud Cover**: spectral group dropout、augmentation、ensemble diversity の考え方を参照する。[official winners](https://drivendata.co/blog/cloud-cover-winners/)
- **Kelp Wanted**: learned auxiliary proxy や multi-scale fusion の使い方を参照するが、外部 imagery や weights は持ち込まない。[official winners](https://drivendata.co/blog/kelp-wanted-winners)
- **STAC Overflow**: external DEM / water data を使う solution は本競技へ転用しない。借りるのは learned gate や residual branch の algorithmic pattern だけである。[official winners](https://drivendata.co/blog/stac-overflow-winners)
- **Local evidence**: `exp029` の append-only IR proxy は fold0 acceptance rule を通らなかった。proxy を再試行するなら append ではなく、低容量の gated residual または diagnostic に限定する。

## 6. 実装候補 inventory

proposal ID は実際の `g_experiments/expXXX` と衝突させないため `XC-XX` を使う。採択後にだけ正式な exp ID を割り当てる。

### XC-01: strict amount × normalized shape

- **Priority**: P0
- **Question**: scalar amount と normalized spatial shape を構造的に分離すると strict OOF tile RMSE が改善するか。
- **Base**: exp035 `no_dilation` の high-resolution architecture だけを参考にする。既存 config / checkpoint は使わず、`context_rows=1`、own-row-only、external-spec canonical mapping / fixed physical coefficients なしの新 config を scratch training する。同一 strict split / seed / preprocessing で比較する。
- **Implementation**:
  - encoder global pooling から positive amount `a` を予測する。
  - decoder から positive `s0` を予測し、pixel mean が 1 になるよう normalize する。
  - final field は `a * s`。
  - main loss は final field の candidate official objective。
  - amount auxiliary は raw amount MSE/Huber と `log1p(a)` 対 `log1p(target_mean)` を別 arm にする。log loss は heavy rain を相対的に弱くするため、main field lossを置き換えない。
  - wet tiles のみ normalized target shape loss を加える。
- **Minimum A/B**:
  1. A: existing final-field head
  2. B1/B2: A + raw / log scalar amount auxiliary
  3. C: full amount × normalized shape
  4. D: C + per-tile candidate loss
- **Diagnostics**: amount RMSE、log-amount RMSE、shape cosine/MSE、wet IoU、positive-pixel RMSE、centroid error。
- **Rule risk**: 上記 strict implementation は Green。target mean は training label から作る auxiliary target である。既存 exp035/exp018 artifact は Amber で、weight initialization、controller、calibration にも使わない。
- **Stop rule**: fold0+4 の両方で方向が揃わなければ full 5-fold を急がない。最終採択は 5-fold で `>=4/5` 同方向かつ平均改善が `0.004--0.005` を超えることを目安にする。

### XC-02: motion-aligned lifecycle hysteresis EDA

- **Priority**: P0。学習前に実行する。
- **Question**: latest cloud state を条件付けた後でも、aligned temporal trend が target amount / wet fraction / upper tail を分けるか。
- **Data boundary**: training rows と各行に含まれる own-row frames のみ。primary は 3 observations の 39,796 rows、robustness は 2 observations の 647 rows、0/1 observations と壊れ frame は missingness group として別集計する。
- **Registration field**: strict 版では satellite ごとに、欠測が少なく frame 間 NCC が安定する channel を **outer-training の入力画像だけ** で選ぶ。外部仕様から IR-window band を固定する arm は amber として分離する。
- **Procedure**:
  1. filename から timestamp を再抽出して必ず昇順 sort し、duplicate / non-monotonic timestamp と broken frame を除外または別群化する。GOES 等の欠損 channel / zero padding を実値と混同しない per-channel valid mask を持つ。
  2. 共通手順で 41×41 にした registration field に対し、連続する過去 frame 間の `dx,dy ∈ [-2,2]` discrete NCC を計算する。`np.roll` の wrap-around は使わず、shift 後に重なる valid 領域だけで評価する。
  3. warp の向きと displacement の符号は人工 translation image の unit test で固定する。
  4. best displacement に加え、`NCC_peak - NCC_zero`、peak-to-second-peak gap を confidence として保存する。gain が 0 以下なら displacement を 0 に戻す。
  5. shift radius `0/1/2/4 px` を比較する場合、入力 NCC だけなら outer-training で選び、target performance で選ぶなら outer-training 内の inner CV を使う。outer-validation に合わせない。
  6. 時刻を `-30/-20/-10 min` と表記できる行では、各 channel `b` について `D1^b = X_-20^b - W(X_-30^b; v1)`、`D2^b = X_-10^b - W(X_-20^b; v2)`、`A^b = D2^b - D1^b` を作る。不規則 interval では `D/Δt` を使い、`A` も derivative の差として定義する。
  7. mean/std/P10/P50/P90、正負 pixel fraction、absolute difference、Sobel/Laplacian roughness change、fold 内 quantile で作る cloud-proxy area change、中央/周辺別統計を保存する。
  8. image-wise min-max normalization は絶対 DN と時間 ordering を壊すため使わない。satellite × channel の median/IQR を outer-training だけで fit する。
  9. latest DN、texture、empirical spectral regime、satellite を continuous features として扱う cross-fitted probe を primary にする。cell 表示は secondary とし、minimum cell count と隣接-bin merge rule を事前固定して過分割を避ける。cell 内 trend は物理名を付けず `D- / stable / D+` とする。
  10. target tile mean、wet fraction、P90/P95、十分 wet な tile の normalized-shape centroid/radius を比較する。
  11. latest-state だけから target amount を予測する cross-fitted probe `g(Z_t)` を作り、residual `target_mean - g(Z_t)` に対して aligned derivative が追加説明力を持つか調べる。最初は同じ Ridge / small GBM で A/B し、画像 model は使わない。
  12. location-held-out 集計と location-cluster bootstrap を行う。
- **Minimum A/B**:
  - A0: latest absolute DN / texture / empirical area
  - A1: A0 + raw Eulerian differences
  - A2: A0 + aligned material-like differences + confidence
  - A3: A0 + raw and aligned differences
- **Negative controls**:
  - N1: same-location × latest-state cell 内で aligned differences を shuffle すると差が消えるか。
  - N1b: location 内 circular time shift または数時間 block permutation で気候差・日周期をなるべく保っても改善が消えるか。
  - N2: estimated shifts の符号を反転するか、別時刻の shift を割り当てると改善が消えるか。
  - N3: latest DN、texture、input-only amount proxy を match した上で、low-confidence scenes では効果が弱く、confidence gate が機能するか。
  - satellite / location 一つだけに依存していないか。
- **Promotion rule**: `Δ = RMSE(A2 or A3) - RMSE(A1)` と定義する。`Δ < 0` が 4/5 folds で再現し、location-cluster paired bootstrap の 95% interval **上端** が 0 未満で、N1/N1b/N2 では改善が消え、outer-training で決めた satellite 内の trend direction が held-out locations で再現したときだけ XC-03 へ進む。satellite 別効果は empirical sign を揃えた後に統合し、未校正 raw DN の符号一致を satellite 間の物理要件にしない。
- **Rule risk**: Green。絶対 wavelength、Kelvin calibration、外部 geometry、他行 frame は使わない。
- **Stop rule**: aligned difference が raw difference を超えない、shuffle でも同程度に改善する、latest-state conditioning 後に効果が消える、confidence と gain が無関係、または outer-training で決めた satellite 内の符号が held-out locations で再現しない場合は棄却する。この場合も「時系列が無価値」と一般化せず、「この時空間解像度では明示的 alignment が raw stack を超えなかった」と限定する。
- **Claim limit**: 合格しても証明できるのは「coarse な latest-state proxy を条件付けても aligned trend に追加予測情報がある」までである。真に同一の雲物理状態、因果的 lifecycle、物理的 hysteresis の証明ではない。

### XC-03: lifecycle-conditioned amount head

- **Priority**: P1。XC-02 合格後だけ実行する。
- **Question**: lifecycle state は主に amount を改善するか。
- **Implementation**: latest global features に aligned trend summary、NCC confidence、displacement、valid mask を加え、amount MLP だけへ入力する。shape decoder への注入は別 arm とする。
- **Minimum A/B**:
  1. XC-01 champion
  2. + unaligned trend
  3. + aligned trend / confidence to amount only
  4. + aligned trend to amount and shape
- **Expected signature**: amount RMSE と high-amount strata が改善し、shape-only diagnostic は大きく変わらない。
- **Rule risk**: Green under own-row-only implementation。
- **Promotion rule**: base に対する 5-fold tile-RMSE delta が `<= -0.004`、かつ 4/5 folds で非悪化を目安とする。改善幅が `0.004` 未満なら、location の大半で同符号かつ cluster interval が 0 未満の場合だけ研究候補として残す。
- **Stop rule**: amount-only arm が base を超えず、amount+shape が一部 fold だけで良い場合は overfit 候補として保留する。ただし複数 folds で centroid / shape diagnostics とともに再現するなら、lifecycle が配置にも効く別機序として再評価する。

### XC-04: feature-level sensor FiLM

- **Priority**: P1
- **Question**: shared representation を保った soft conditioning は unseen-location generalization を改善するか。
- **Condition variables**: provided satellite identity、own-row frame count / mask、提供 metadata だけから得る時刻区分。外部 ephemeris や solar geometry は使わない。
- **Implementation**: small embedding MLP から各 encoder stage の channel scale / bias を出す。output head は共有し、FiLM を identity 付近で初期化する。
- **Minimum A/B**: none、satellite only、availability only、satellite+availability。location ID は入れない。
- **Local distinction**: `exp032` の hard output separation と異なり、parameter sharing を維持する。
- **Stop rule**: satellite 内 OOF だけ改善し outer-location が悪化する、または一 satellite のみで全 gain が生じるなら採択しない。

### XC-05: shared temporal encoder + feature differences

- **Priority**: P1
- **Question**: raw channel concat より shared per-frame encoding が temporal state を安定して抽出できるか。
- **Implementation**: 同一 encoder を各 own-row frame に適用し、latest feature、past feature、signed difference、absolute difference、valid/age embedding を scale ごとに fuse する。
- **A/B**: exp028-style raw latest/`|Δ|` concat vs shared encoder fusion。parameter count を可能な限り合わせる。
- **Rule risk**: Green if frames remain within the row。
- **Stop rule**: improvement が missing-frame strata に限定され、全体 noise floor 未満なら XC-11 とまとめて低優先にする。

### XC-06: nested rain-threshold shape auxiliaries

- **Priority**: P1
- **Question**: footprint と intensity contours を明示的に学ぶと shape localization が改善するか。
- **Targets**: `y > τ_k` の nested masks。`τ_k` は training fold の quantiles か事前固定した rain-rate thresholds から選び、validation label で調整しない。
- **Implementation**: shared decoder から 3--4 binary logits と optional Sobel-gradient head を出す。weight は低くし、continuous field loss を主とする。
- **A/B**: none、wet mask only、nested masks、nested masks+gradient。
- **Inference**: auxiliary predictions は捨てる。hard mask postprocess は別実験にする。
- **Stop rule**: IoU だけ上がり continuous tile RMSE が改善しない場合は採択しない。

### XC-07: regime-stratified sampler with correction

- **Priority**: P2
- **Question**: rare heavy/wet tiles の gradient exposure を増やしても calibration を保てるか。
- **Strata**: satellite × target-mean bin × wet-fraction bin。bin edges は fold training data のみで決める。
- **Implementation**: batch 内の最低 quota を設け、loss には sampling probability の inverse または tempered correction を入れる。
- **Validation**: validation distribution は変更しない。
- **Stop rule**: high-rain RMSE が改善しても dry/mid-range の悪化で total tile RMSE が相殺されれば終了する。

### XC-08: distributional tile amount

- **Priority**: P2
- **Question**: fat-tailed amount を bins / ordinal distribution として学ぶと deterministic regression より安定するか。
- **Implementation**: training-fold edges の 32 / 64 bins、ordinal または categorical head、期待値 serving。continuous amount head との multi-task arm も作る。
- **Calibration checks**: OOF reliability、CDF collapse、upper-bin saturation、expected amount bias、satellite別 calibration。
- **A/B**: deterministic、unweighted categorical、tempered-frequency categorical、multi-task。
- **Stop rule**: OOF distribution が良くても expected-value tile RMSE が XC-01 champion を超えなければ submission へ使わない。

### XC-09: registered shape auxiliary

- **Priority**: P2
- **Question**: small residual displacement を許容した shape learning が exact field を改善するか。
- **Implementation**: exact `y_hat` lossを保持し、normalized shapes の `±1`、必要なら `±2` pixel shifted losses の soft minimum を低 weight で加える。
- **A/B**: no auxiliary、±1、±2。shift range と weight を同時に増やさない。
- **Stop rule**: exact metric が改善しない、centroid bias が増える、satellite ごとの preferred shift が不安定なら終了する。

### XC-10: HRNet-lite / high-resolution branch

- **Priority**: P2
- **Question**: amount を維持したまま fine shape を改善できるか。
- **Implementation**: 41×41 native-resolution branch を保持し、low-resolution semantic branch と反復 fusion する。parameter / FLOPs を記録する。
- **A/B**: XC-01 champion decoder vs matched-capacity HRNet-lite。
- **Stop rule**: positive-pixel / boundary diagnostics だけ改善し total tile RMSE が noise floor 未満なら計算コストに見合わない。

### XC-11: age-aware frame drop / repeat

- **Priority**: P3
- **Question**: variable availability に頑健か。
- **Implementation**: training 中だけ own-row past frame を drop または duplicate し、valid、duplicate、relative-age masks を必ず入力する。
- **Caveat**: 39,796 / 40,686 rows が 3 observations なので、実分布から大きく外れる augmentation を多用しない。
- **Stop rule**: complete-frame majority の performance が悪化するなら終了する。

### XC-12: conditional band permutation and band-group dropout

- **Priority**: P2 EDA / P3 training
- **Question**: satellite ごとに、どの channel group が amount、shape、lifecycle に寄与するか。
- **EDA**: validation 内で satellite と粗い cloud-state strata を保ちながら band/group を permutation し、amount / shape diagnostics の変化を分ける。
- **Training**: redundant group だけを低確率で drop し、missing mask を付ける。
- **Rule risk**: Green if all statistics are fit inside each training fold。
- **Stop rule**: importance ranking が fold / satellite で反転する場合、固定 band selection は行わない。

## 7. 共通評価 protocol

### 7.1 Split と artifact hygiene

1. strict current-row-only data loader を固定する。
2. fold split、seed、epoch budget、augmentation、checkpoint selection を A/B で揃える。
3. fold 内で学ぶ quantile、bin edge、normalization、alignment sign を validation や全 OOF へ leak させない。
4. amber successor-context checkpoint、controller、blend weight を green experiment に混ぜない。
5. fold0+4 を screening に使ってよいが、採択表には 5-fold fixed configuration だけを載せる。
6. calibration、controller、blend はすべての upstream artifact の最も厳しい risk を継承する。strict OOF で outer cross-fit したものだけを Green とする。

### 7.2 Primary / secondary metrics

| Level | Metric | 目的 |
| --- | --- | --- |
| Primary | mean per-tile RMSE | leaderboard と最も整合するローカル指標 |
| Secondary | pooled pixel RMSE | metric ambiguity と大域誤差の監視 |
| Amount | RMSE/MAE of tile mean、log1p amount RMSE | XC-01/03 の機序確認 |
| Shape | normalized-shape MSE/cosine、wet IoU、nested IoU | localization の機序確認 |
| Positive | wet-pixel RMSE、P90/P95 bias | heavy-rain degradation の監視 |
| Geometry | centroid distance、gradient error | displacement/boundary の監視 |
| Calibration | amount reliability / bin occupancy | XC-08 の collapse 検出 |

### 7.3 Required stratification

- satellite
- outer location
- provided time/day-night proxy（規約確認済み metadata のみ）
- own-row frame count
- target amount decile
- wet fraction decile
- lifecycle state and alignment confidence（XC-02 合格後）

### 7.4 Acceptance gate

- 事前に primary metric と champion comparator を固定する。
- 5 folds 中 4 folds 以上で改善方向が一致することを目安にする。
- mean gain が経験的 noise floor `0.004--0.005` を超えることを strong promotion の目安にする。
- location-cluster bootstrap interval と worst-location degradation を併記する。
- 多数の arm から最良値だけを選ばず、arm 数と selection path を記録する。
- paper candidate は機序 diagnostic も仮説どおりに変化することを要求する。スコアだけの改善は competition engineering としては有用でも、機序の証拠にはしない。

## 8. 規約・license gate

最終判断は [competition_rules.md](competition_rules.md) と運営回答を優先する。本節は研究案を安全側に分類するための実務 gate である。

### 8.1 Green

- 同一 CSV row に明示された satellite observations のみを使う。
- training target から fold 内で auxiliary target、threshold、bin、normalization を作る。
- source-only domain generalization、frame dropout、mixup、FiLM を行う。
- 外部論文の一般概念を clean-room で再実装する。
- 公式の許可 license 条件を満たし、URL/version/license/取得日/SHA-256/load 箇所を manifest 化した公開 program / pretrained model / weight を、運営規約の範囲で使う。
- model selection と calibration を training folds / OOF の範囲で行う。

### 8.2 Amber: 書面回答または artifact 分離が必要

- successor row / cross-row temporal context
- evaluation images を使う self-supervision、test-time adaptation、pseudo-labeling
- 外部の wavelength 数値、spectral response function、radiance-temperature LUT、satellite geometry、parallax constants
- GeoNames / Nominatim 等で取得した座標由来特徴は、公式 permalink、取得日、license と運営判断の証跡が揃うまで green にしない
- license、base model、training-data provenance、commit、hash のいずれかが未確認の code / checkpoint。Amber で許すのは metadata 監査だけとし、license 確認まで code copy / execution / weight load を行わない。未解決なら使用しない

探索する場合も、green champion の checkpoint、OOF、submission と完全に分離する。

### 8.3 Red: 本競技 pipeline に入れない

- 外部 rain gauge、radar、NWP、DEM、IMERG、GPM DPR/CORRA 等の値を feature / target / calibration に使用する。
- 外部 dataset 由来の normalization statistics、LUT、climatology を直接 feature preprocessing に使用する。
- train target overlap を evaluation prediction へコピーする。
- evaluation target の proxy、placeholder、推定値を label として学習する。
- 他コンペの test pseudo-label や external EO imagery を持ち込む。
- 禁止された公開データや private data で学習した model を無監査で使う。
- 不許可 license、権利不明、private-only の code / weight を使う。

evaluation image から自モデルだけで作る self-pseudo-label / TTA は transductive **Amber**、placeholder target、外部 target proxy、overlap/reverse-engineering から作る推定 label は **Red** と区別する。

### 8.4 Code / checkpoint manifest

外部実装を参照または使用する場合、最低限次を記録する。

```text
source URL
paper / repository title
commit or released version
retrieved date
license text and scope
checkpoint URL and training-data provenance
file SHA-256
copied code paths or clean-room declaration
competition-rule decision and reviewer
```

論文を読んで概念だけを再実装する場合も citation を残す。repository に license が見当たらない場合、code をコピーせず concept-only に留める。checkpoint card の license と code repository の license は別々に監査する。

### 8.5 Geocoding discussion の扱い

ローカル和訳 [approved_geocoding_sources_ja.md](../discussion/approved_geocoding_sources_ja.md) と [geocoding_coordinates_ja.md](../discussion/geocoding_coordinates_ja.md) には、GeoNames と Nominatim/OpenStreetMap を承認 source とし、EPSG:4326 座標および位置・時刻の閉形式関数を許可する運営回答が整理されている。一方、現在のローカル記録には公式 topic/message permalink、投稿日、回答 checksum が残っていない。

したがって [competition_rules.md](competition_rules.md) の最新判断に従い、座標・local solar time・solar geometry は **amber→green candidate** とする。最終 model へ入れる前に公式証跡を回収し、source record ID/query URL、canonical name、lat/lon、取得日、変換 script を固定する。標高、海岸線距離、気候区分、気候値、土地被覆は、無料 source であっても外部 data join なので使用しない。

この geocoding 例外は successor-row context、evaluation adaptation、overlap target copy を許可するものではない。また位置を入れる場合も、高容量の location memorization ではなく低容量の物理 conditioning とし、outer-location CV で判定する。

## 9. 論文化可能な研究仮説

### 9.1 主仮説

**H1: Amount--shape factorization**

未見 location への satellite precipitation retrieval では、global tile amount と normalized spatial shape を明示的に分けると、単一 dense regression head より calibration と localization の trade-off が改善する。

必要な証拠:

- XC-01 の strict 5-fold improvement
- amount diagnostic の改善
- shape diagnostic を悪化させないこと
- satellite / location strata にまたがる再現
- simple scalar auxiliary との isolated ablation

**H2: Lifecycle-conditioned hysteresis**

現在の cloud-top appearance を条件付けても、motion-aligned temporal trend は IMERG target amount を分ける。

必要な証拠:

- XC-02 の within-state conditional separation
- shuffle negative control で差が消えること
- unaligned より aligned trend が有効であること
- outer-location で符号と大きさが再現すること
- XC-03 で amount error が選択的に改善すること

**H3: Soft sensor conditioning**

hard satellite-specific output heads ではなく、shared feature representation への低容量 conditioning が cross-sensor / unseen-location generalization を改善する。

必要な証拠:

- XC-04 が no-conditioning と hard-head historical result を上回る
- gain が単一 satellite に偏らない
- location ID を用いず再現する

### 9.2 論文候補

> Lifecycle-conditioned Amount--Shape Factorization for Cross-Sensor Geostationary Satellite Precipitation Retrieval under Unseen Locations

competition phase の主張範囲は「IMERG target retrieval」とする。real precipitation skill を主張する post-competition extension では、GPM DPR / CORRA 等の独立 reference を用いる研究を別 repository、別 data manifest、別 experiment table として実施し、competition submission pipeline と混ぜない。

### 9.3 Novelty を守る比較

- dense regression vs scalar auxiliary vs full factorization
- raw trend vs aligned trend vs shuffled trend
- no conditioning vs feature FiLM vs hard output heads
- exact loss vs exact + registered shape auxiliary
- random-row CV summaryだけでなく outer-location analysis

外部コンペ手法を並べただけでは novelty にならない。本研究の中心は、`amount--shape--lifecycle` の因果的に解釈可能な分解、strict unseen-location protocol、product-target という限界を含む検証設計に置く。

## 10. 実行順と停止判断

### Phase A: 即時

1. exp035 `no_dilation` の残り folds を完了し、Amber pipeline 内で architecture axis を閉じる。
2. `context_rows=1`、外部仕様 mapping なし、scratch training の strict base を新規に確立する。
3. strict base 上で XC-01 の arms を同一 code path で実装する。
4. XC-02 を `g_eda` として実装し、model training と独立に判定する。

### Phase B: evidence-driven

1. XC-02 合格時だけ XC-03 を実装する。
2. XC-04 と XC-05 は XC-01 champion 上で一軸ずつ試す。
3. XC-06 は shape residual が残る場合に追加する。

### Phase C: orthogonal extensions

XC-07/08/09/10/11/12 を、primary failure mode に応じて選ぶ。全てを網羅的に走らせない。

### 現時点で再優先化しない案

- 追加の dilation
- D4 / rotation TTA
- hard satellite-specific output heads
- prediction-field の advected smoothing
- append-only single IR proxy
- GAN / diffusion を final RMSE field の主生成器にすること
- importance correction のない強い oversampling
- external foundation weights / external data
- evaluation pseudo-labeling

## 11. 実験 ticket に必要な記録

各 XC を正式 exp に昇格するとき、次を ticket と manifest に固定する。

- hypothesis と causal mechanism
- source claim / local evidence / transfer inference
- strict/amber risk label
- exact data boundary
- comparator、folds、seed、budget
- primary metric と acceptance threshold
- required diagnostics と negative controls
- stop rule
- code commit、config hash、container image
- output/checkpoint directory
- result by fold / satellite / location
- decision: adopt、park、close

## 12. 主要資料

### Local

- [Research survey Round 3](research_survey_v3_2026-07-16.md)
- [Competition rules](competition_rules.md)
- [Experiment plan](../g_experiments/EXPERIMENT_PLAN.md)
- [Exact factorization EDA report](../outputs/g_eda/exp006/EDA_REPORT.md)
- [Round 5 experiment plan](plan/round5_experiment_plan_2026-07-16.md)
- [Approved geocoding sources: local Japanese record](../discussion/approved_geocoding_sources_ja.md)
- [Geocoding coordinates discussion: local Japanese record](../discussion/geocoding_coordinates_ja.md)

### Physics / product

- [JMA: CCI and RDCA](https://www.data.jma.go.jp/mscweb/technotes/msctechrep62-2.pdf)
- [Semi-Lagrangian deep convective cloud tracking](https://amt.copernicus.org/articles/16/1043/2023/)
- [SATCAST object tracking](https://journals.ametsoc.org/view/journals/apme/51/11/jamc-d-11-0246.1.xml)
- [Cloud-type-dependent IR--rain relationship](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.3288)
- [PDIR](https://pmc.ncbi.nlm.nih.gov/articles/PMC8216223/)
- [NASA IMERG V07 documentation](https://gpm.nasa.gov/resources/documents/imerg-v07-technical-documentation)
- [IMERG V07 ATBD](https://gpm.nasa.gov/sites/default/files/2023-07/IMERG_V07_ATBD_final_230712.pdf)
- [CMORPH](https://www.ftp.cpc.ncep.noaa.gov/precip/CMORPH_V1.0/REF/Joyce_et_al_2004_JHM_CMORPHP.pdf)
- [NowcastNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC10356617/)
- [JMA AHI bands](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html)

### Competition / solution reports

- [Weather4Cast 2023](https://weather4cast.net/neurips2023/)
- [Weather4Cast official challenge report](https://proceedings.mlr.press/v220/gruca23a.html)
- [Open Cities official winners](https://drivendata.co/blog/open-cities-disaster-winners/)
- [DrivenData competition winner code index](https://github.com/drivendataorg/competition-winners)
- [xView2 multi-temporal fusion](https://arxiv.org/abs/2004.05525)
- [PROBA-V HighRes-net](https://arxiv.org/abs/2002.06460)
- [SpaceNet 7 report](https://arxiv.org/abs/2102.11958)
- [SpaceNet 4 organizer analysis](https://medium.com/the-downlinq/a-deep-dive-into-the-spacenet-4-winning-algorithms-8d611a5dfe25)
- [NTIRE 2021 burst super-resolution](https://openaccess.thecvf.com/content/CVPR2021W/NTIRE/html/Bhat_NTIRE_2021_Challenge_on_Burst_Super-Resolution_Methods_and_Results_CVPRW_2021_paper.html)
- [Data Science Bowl nuclei report](https://www.nature.com/articles/s41592-019-0612-7)

## 13. 最終判断

外部コンペから最も価値があるのは、特定 backbone のコピーではなく、**問題を適切な中間変数へ分け、各変数に対応する negative control を置く設計** である。本課題では、その中間変数が amount、shape、lifecycle、sensor state である。

優勝に直結する第一歩は XC-01 の strict A/B、論文化に直結する第一歩は XC-02 の conditional hysteresis test である。この二つを同時並行で進め、スコア改善と科学的機序の両方を 5-fold / outer-location evidence で結ぶ。
