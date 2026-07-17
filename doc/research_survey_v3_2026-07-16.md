# 研究調査 Round 3: 勝つための安全な解法と論文化可能な仮説

最終更新: 2026-07-16 UTC

## 0. この文書の目的と証拠階層

この文書は、2026-07-16時点の `doc/`、`discussion/`、OOF、Public LB、Slurmログを
再監査し、衛星降水推定・IMERG・Himawari/GOES/Meteosatの一次資料と接続したものです。
狙いは次の三つです。

1. 現在のスコア改善を、規約上安全な最終提出へ変換する。
2. 残差の支配要因に対して、期限内に勝率の高い実験へ集中する。
3. コンペのtargetを当てる工学と、実降水を推定する科学を区別し、論文化できる
   falsifiable hypothesisを作る。

判断の優先順位は、(A) 公式コンペページ・公式回答、(B) 手元のOOF/LB/実ログ、
(C) 衛星機関の技術文書、(D) 査読論文・preprintです。論文や外部productは設計知識として
のみ参照し、コンペ用pipelineへ外部の値・画像・統計を持ち込みません。

## 1. 結論

### 1.1 最も大きい技術的更新

現時点で最も高レバレッジな未解決候補は、**タイル面平均降水強度のcalibration**です。
exp018の5-fold OOF `tile_rmse=0.6093` に対し、正解のtile meanへ予測場をrescaleする
`amount_swap` は`0.5446`、正解wet mask上へ予測fieldのpixel sumを一様配置する`mask_swap`は
`0.7111`でした。前者は約`0.0647`の大きなcounterfactual改善を示します。

ただし、これはamountとplacementの寄与率分解ではありません。`amount_swap`のmean一致scaleは
固定shapeに対するL2最適scaleとは限らず、予測meanがほぼ0の行ではflat fieldへ置換されます。
`mask_swap`も真のmask内強度shapeを破壊します。証明できるのは「amount calibrationが有望」
までで、「位置より量が支配」と断定するには、fieldをtile meanとzero-mean spatial residualへ
直交分解する追加診断が必要です。その上で次の本命を、**面平均強度と正規化空間形状を明示分解
して同時学習するモデル**とします。

追加診断では`r = y - mean(y)`、`r_hat = y_hat - mean(y_hat)`として、各tileの
`MSE = (mean(y_hat)-mean(y))^2 + mean((r_hat-r)^2)`を用います。これならmean errorと
zero-mean spatial residual errorを交差項なしに測れます。この恒等式はtile単位の診断です。
最終比較では両成分を保存し、確認後のserver aggregation（全画素poolまたはtile norm平均）を
それぞれ再構成します。分解だけから未確定の集約法を仮定しません。

### 1.2 最も大きい科学的更新

公式ページのsource linkはIMERG V07を指しますが、競技targetのrun
（Early/Late/Final）と詳細provenanceは未同定です。一般的なIMERG V07生成系は、PMW retrievalを
GPROFで処理し、CMORPH-KFによる準Lagrangian補間、PDIRのgeo-IR降水推定、補助解析場等を
組み合わせて30分値を作ります。各画素・時刻のIR寄与はPMW availability等で変わります。
したがって、次の二つが混在する可能性を仮説として検証します。

- 雲画像から実降水を推定する問題
- geo-IRも内部利用するIMERGの生成特性を模倣する問題

単純な予測時間平滑化がPublicで約`-0.0045`効いたことも、真の降水物理だけでなくIMERGの
時空間生成過程に整合した可能性があります。target run/provenanceを確認するまでは原因と断定しません。
[NASA IMERG V07 technical documentation](https://gpm.nasa.gov/resources/documents/imerg-v07-technical-documentation)

### 1.3 最も大きい規約上の更新

現Public best `0.6661746682` は、overlap copy patch、successor row、隣接行の両方向
平滑化を含む**探索スコア**です。公式規約には、すでに存在する正解の利用やreverse
engineeringではなく、実世界で適用可能なモデルを期待する旨があります。このため、
train targetをevaluationへコピーするoverlap patchは最終提出から外します。
successor rowとevaluation横断平滑化も、運営の書面回答までは最終本線から分離します。

一方、ローカル文書にあった「外部pretrained weight全面禁止」は正しくありません。
公式ページが列挙するライセンスの公開pretrained model/weightは条件付きで利用可能です。
ただし、出典、checkpoint、ライセンス、取得日、hashを完全に残し、有料・排他的・
非商用限定のものは使いません。
[公式コンペページ](https://community.solafune.com/competitions/f87811b8-1964-4f4b-84b3-6fddd67ec4b1)

### 1.4 論文化の本命

期限内の本命は次の統合仮説です。

> 少数の未見地点へ汎化する量子化済み多衛星QPEでは、センサー別hard expertより、
> 配布band IDからtrain内で学習する量子化頑健なsoft spectral-regime表現と、
> 面平均強度・正規化shapeの分離、
> 候補評価ノルムに整合した最終学習を組み合わせる方が頑健である。

コンペ後の科学論文では、IMERGをdense auxiliary teacherに留め、DPR/CORRA等を主教師に
した別データ系で「IMERG模倣」と「実降水推定」を分離検証する必要があります。外部データを
使うこの研究系は、コンペ用workspace・重み・統計と完全に隔離します。

## 2. 現在地の再構成

### 2.1 データと検証上の制約

- train 40,686行、evaluation 29,090行、出力は41×41。
- train/evaluationの地点は20/18で重複なし。
- trainは20 locations、28 location-month blocksで、行間の時間相関が強い。有効標本数は
  40,686より小さいが、独立な気象episode数は未推定。
- row-level random splitは禁止。location単位GroupKFoldを使う。
- evaluation `test_files/` はplaceholderであり、値や統計を一切利用しない。
- 最新の修正版datasetであることを配布物checksumまたは再取得記録で確定する。

### 2.2 スコアの節目

| 段階 | Public RMSE | 技術 | 最終規約状態 |
| --- | ---: | --- | --- |
| exp001 | 0.75320 | 54ch Compact U-Net | green |
| exp004 | 0.72525 | rain occurrence + amount | green |
| exp009 | 0.71534 | successor row | amber |
| exp015 | 0.70967 | OOF isotonic calibration | amber（入力がexp009） |
| exp014 | 0.69687 | overlap copy patch | red |
| exp016 | 0.69776 | hurdle log-normal | amber（successor） |
| exp017 | 0.69974 | 波長整列・物理特徴 | amber（successor） |
| exp018 | 0.69295 | high-resolution localizer | amber（successor） |
| exp024 | 0.69193 | exp016/017 blend | amber（successor） |
| exp036 smooth | **0.66617** | blend + blur + threshold + patch + 3-tap cross-row smoothing | red/amber混在 |
| exp037 | 0.66626 | exp036 stack + 6-view rotation/flip TTA | red/amber混在 |

現行 `0.66617` と1位snapshot `0.63474` の差は約`0.03143`です。ただし比較すべき
「strict champion」は、CSVの当該rowだけを読む既知のexp011 `0.72323`です。これは
能力差ではなく、規約境界の違うスコアを混ぜないための管理上のanchorです。

### 2.3 何が効き、何が閉じたか

効いたものは、successor row、overlap patch、isotonic、hurdle head、高解像局在化、
異種model blend、衛星別weight、blur/threshold、時間平滑化です。閉じたものは、複雑な
temporal fusion、単純satellite adapter、cleanup、弱いseedの等重みblend、単一IR proxy、
focal/Tversky、GPM 0.01 snap、rotation TTAです。exp032のconditional satellite headは
Meteosat-only fold0で悪化し、exp035の`exp028 inputs + dilated bottleneck`統合armはfold0/4で
悪化しましたが、これだけでhard expert一般やdilation単独を棄却したとは扱いません。

したがって次は、失敗済みの案を名前だけ変えて反復しません。

- hard satellite expertではなくshared representation + soft regime
- raw optical flow vectorではなくmotion-compensated cooling residual
- 画素amount headの微修正ではなくtile mean intensity × normalized shape
- 予測をそのまま移流平均するのでなく、入力の成長・冷却signalを整列して抽出

### 2.4 最新のnegative resultも保持する

GPM予測場を予測場自身のshiftで合わせるadvected smoothing再検証は、static
`0.59334`に対しadvected `0.59346`（`+0.00012`）で悪化しました。よって予測レベルの
移流平滑化は棄却します。ただし、衛星入力を整列した後のcloud-top coolingは別仮説であり、
このnegative resultからは棄却されません。

## 3. ルール監査: green / amber / red

### 3.1 Green: 最終提出に使える

- 配布trainの衛星GeoTIFF、target、CSVだけで学習する。
- 各evaluation rowに明示された観測ファイルだけで推論する。
- 配布trainからfitした正規化、経験的band contrast、texture、flow、補助task、OOF calibration。
- 競技ページが提供するband名・indexの直接利用と、配布trainだけで学ぶsensor別band projection。
- location GroupKFoldと外側cross-fittingでfit・評価したensemble/controller。ただし、すべての
  source predictionがstrict-row-onlyである場合に限る。
- 公式discussion permalink・topic/message ID・取得日まで証跡化したGeoNamesまたはNominatimの
  座標と、その座標から閉形式で計算するlocal solar time / solar geometry。source ID、query、
  座標、取得日、scriptも保存する。
- 公式に許可されたライセンスの公開pretrained weight。checkpoint単位で出典・license・hashを
  保存し、最終コードから再取得可能にする。
- 公開論文・衛星機関文書を、モデル設計の知識として読む。

### 3.2 Amber: 書面回答が得られるまで探索限定

- evaluationのsuccessor rowから対象時刻以後の衛星画像を読む。
- evaluationの複数rowを読むtemporal filter/postprocess。causal/bidirectional、3/5-tapを問わない。
- evaluation画像を使うtransductive/self-supervised adaptation。
- 外部仕様から推定したsensor間bandカテゴリ対応、衛星直下点、数値中心波長/SRF、校正式、
  固定radiometric coefficientを特徴計算へ埋め込む。
- 許可license自体は確認済みだが、checkpoint由来・配布元・再現証跡が未確定のpretrained weight。
- 現在のローカル和訳に欠けるgeocoding回答の公式permalink・topic/message IDを未保存のまま、
  coordinate-derived featureを最終候補へ昇格する。

Amberは、提出zip、実験ID、OOF、重み、後処理をgreen系と混ぜません。運営回答が得られない
場合、最終提出には使いません。blend、calibration、controller、fine-tune等の派生artifactは、
すべてのupstreamのうち最も厳しいrisk区分を継承します。

### 3.3 Red: 最終提出に使わない

- train GPM targetを空間重複するevaluation画素へコピーするoverlap patch。
- 外部Himawari HSD/L1/L2、GOES ABI、MTG FCI、追加IMERG、GSMaP、PDIR、CORRA、
  DPR、NWP、再解析、DEM、海岸線、土地被覆、気候統計を取得・結合する。
- 未承認geocoderや外部座標表、外部product由来の平均・分散・LUT・calibration係数。
- evaluation placeholder targetの値や統計を使う。
- コンペ中に配布データやsecondary productを公開する。
- license・権利が不明、または許可licenseと非互換な実装・weightを利用・実行する。

「submission formatがvalid」は手法の規約承認ではありません。

### 3.4 直ちに運営へ確認する質問

1. hidden evaluatorは全TIFFの全画素をpoolしたRMSEか、TIFFごとのRMSEの平均か。
2. 各rowに列挙されていない、同一locationの後続rowに含まれる衛星ファイルを利用できるか。
3. evaluationの複数rowを読むtemporal filter・joint postprocessingは、過去方向だけの場合も
   含めて許可されるか。
4. labelを使わないevaluation画像でのself-supervised adaptationは許可されるか。
5. 公式衛星仕様から推定するsensor間bandカテゴリmapping、直下点、中心波長、SRF等を
   特徴生成に使えるか。
6. 許可pretrained weightについて、必要なlicense証跡と学習データ条件は何か。

質問2–4は「配布物だけだから可」と自己判断しません。質問1はモデルの最適推定量そのものを
変えるため、最優先です。

## 4. 指標監査: 重要仮説だが、まず定義を確定する

### 4.1 公式表示と実測の矛盾

公式ページと公式utilityの一般的なRMSE実装は、渡された配列の全要素をpoolして
`sqrt(mean(error^2))`を返します。ただし、serverがutilityを全submissionへ一度呼ぶか、
fileごとに呼んで平均するかはutilityだけでは分かりません。一方、ローカルの標準は、各41×41
tileのRMSEを先に計算してsample平均する`mean_i sqrt(mean_p(error_i,p^2))`です。
[Solafune tools](https://github.com/Solafune-Inc/solafune-tools/blob/main/solafune_tools/metrics.py)

primary analysisは、OOFとPublicが同じmodel出力に対応するpure-model 10件です。exp008/015を
加えた12行版では、postprocess後OOFがなく、それぞれexp004/009のraw OOFを再利用するため
heuristicとしてのみ扱います。

| subset / OOF predictor | PublicとのSpearman | linear-fit residual std |
| --- | ---: | ---: |
| matched pure models 10件 / tile平均 | **0.964** | **0.0041** |
| matched pure models 10件 / 全画素pool | 0.636 | 0.0143 |
| heuristic 12行 / tile平均 | 0.961 | 0.0041 |
| heuristic 12行 / 全画素pool | 0.681 | 0.0134 |

結果はtile平均と整合的ですが、10件は近縁modelで、checkpointや実験判断の一部もtile指標に
基づくためselection biasがあります。OOFとPublicのlocation/distribution shiftも交絡します。
したがってserver内部の集約方法を証明せず、独立な提出対または運営回答で確定します。
再現表は `outputs/l_eda/exp003/CV_LB_CALIBRATION.md` にあります。

### 4.2 数学的帰結

- 全submissionを一度だけpoolする固定評価集合では、RMSEとSSEの順位は同じです。通常の
  population MSE riskまたは大標本近似の下では、pointwise Bayes actは条件付き平均です。
  ただし、ランダムな有限評価集合に対する`E[sqrt(mean(error^2))]`を直接最小化する別の
  decision problemでは、rootのため厳密には条件付き平均と限りません。
- serverが等重みのtile normを平均する場合、有限一次モーメントを持つ条件付き分布と制約なしの
  population riskの下で、Bayes-optimal fieldは41×41次元の**条件付き空間中央値
  （geometric/spatial median）**です。解は非一意の場合があります。

この違いは単なるloss名の差ではありません。既存の「RMSEだから常にconditional mean」や、
既知tile内の最良な平坦値を求めるoracleからは結論できません。
[Chaudhuri 1996](https://doi.org/10.1080/01621459.1996.10476954),
[Vardi & Zhang 2000](https://pubmed.ncbi.nlm.nih.gov/10677477/)

### 4.3 指標確定後の低コスト実験

1. strict-row-onlyに再学習したmodel群のOOF fieldで、算術平均、画素中央値、Weiszfeld
   geometric median、outer-cross-fit weighted geometric medianを比較する。既存exp016/017/018を
   そのまま使う版はsuccessor由来のamberである。
2. strict base checkpointのfield reconstruction項を
   `mean_batch(sqrt(mean_pixels(error^2)+eps))`で短くfine-tuneし、再blendする。これは
   `eps>0`では近似lossなので、target scaleに応じたeps sweepと`eps=0`のexact lossを比較し、
   batch全体に対してrootを取らない。
3. 複数軽量headを学習する場合はEnergy Scoreも比較し、最終fieldを幾何学的中央値で
   collapseする。

既存modelのgeometric median ensembleは条件付き空間中央値の厳密推定ではなく、まず安価な
heuristicです。複数headを離散分布のatomと明示するか、Monte Carlo sampleとして学習するかを
区別し、単なるdeterministic headを自動的に確率sampleとは呼びません。1681次元Energy Scoreは
空間依存の誤指定への識別力が弱いため、variogram scoreまたは低次元projection診断も併用します。
[Gneiting & Raftery 2007](https://doi.org/10.1198/016214506000001437)

## 5. 優先仮説

| 優先 | 仮説 | 勝率 | 論文新規性 | 規約 | 実装量 |
| --- | --- | --- | --- | --- | --- |
| P0 | metric definition確定 + direct loss/field median | 中（低コスト） | 候補 | strict sourceならgreen | 小 |
| P0 | tile mean intensity × normalized shape | **最も高い候補** | 候補 | green | 中 |
| P0 | strict causal exp018-family baselineの新設 | 高 | 低 | green | 中 |
| P1 | quantization-aware empirical spectral soft regime | 中–高 | 高候補 | train-learned mappingはgreen、外部仕様mappingはamber | 中 |
| P1 | motion-compensated cloud-top cooling | 中 | 高 | green | 中 |
| P1 | regime-conditioned OOF ensemble controller | 中 | 中 | strict sourceならgreen | 小–中 |
| P2 | train-only future-satellite auxiliary task / masked pretraining | 中 | 中–高 | green | 中–大 |
| P2 | constrained learned displacement correction | 中 | 高候補 | amber/green化要確認 | 中 |
| P3 | normalized shape residual diffusion/generative model | 低 | 中 | green | 大 |

### 5.1 P0: Tile mean intensity × normalized shape

まず確率的意味を持たないdeterministic parameterizationとして、予測を次のように分けます。

```text
m       = nonnegative tile-mean intensity head
z       = nonnegative spatial activation g(S)
s       = z / clamp_min(mean_pixels(z), eps)
y_hat   = m * s
```

`mean_pixels(z)>=eps`なら`mean_pixels(y_hat)=m`となり、intensity headが面平均強度を
直接担当します。shapeは尺度から解放され、雲頂構造・rain mask・局在化に集中できます。
`g`はsoftplusだけでなく、空間的な厳密な0を表現できるsparse nonnegative activationも比較します。
後段thresholdを使えばmean identityが崩れるため、threshold後の再正規化もablationします。

別案として`m=q*a`、`q=P(wet|X)`、`a`をwet時のpositive intensityとするhurdle補助headを
置けます。ただし、一般にamountとshapeは条件付き独立ではなく、tile-normのBayes actのmeanが
`q*E[a|wet]`になる保証もありません。direct field lossと複数lossを併用した後の`q/a`を厳密な
Bayes functionalとは呼ばず、BCEとpositive-only lossで意味を固定した補助headとして扱います。

損失候補は次です。

- tile wet BCE
- unconditional `m`のMSEまたはTweedie deviance
- hurdle版positive `a`のlog-Huber、Gamma、またはlog-normal NLL
- normalized target shapeへのcosine / normalized MSE / soft Dice
- 再構成fieldへのcandidate metric loss
- 41→20→10等のmultiscale mean-intensity consistency

Tweedieはunconditional tile meanのゼロ質量と正のheavy tailをcompound Poisson–Gammaで
一体化できますが、別の`q`と併用するとゼロを二重にmodel化します。hurdle版のpositive側に
標準Tweedieをそのまま置くとゼロ質量を扱う利点は消え、厳密にはzero-truncated familyの検討が
必要です。固定index parameterのTweedie devianceは条件付き平均をelicitationしますが、完全な
確率modelにはdispersionも必要です。index選択はfold内で行い、この引用は査読済み結論ではなく
2025年のpreprintとして扱います。
[Stop Using RMSE as a Precipitation Target](https://arxiv.org/abs/2509.08369)

log-normal NLLで`a`を条件付き期待値としてserveする場合は`exp(mu+sigma^2/2)`を使います。
dry tileではtarget shapeが未定義なのでshape lossをskipし、微小amountも下限thresholdで除外します。
最小ablationは、(A) current pixel hurdle、(B) tile mean head追加、(C) mean-shape完全分離、
(D) C + candidate metric lossです。`amount_swap`の約`0.0647`は期待改善値でも数学的上限でも
ありません。

### 5.2 P1: Quantization-aware Empirical Spectral Regime Mixture

strict-green版では、競技ページに与えられた16個のband名・indexをcategorical IDとして使い、
sensor別の小さな1x1 projectionから共有latent band tokenへの対応を配布trainだけで学習します。
外部仕様による物理カテゴリをhard-codeせず、sensor-shared soft regimeも教師なしlatent gateとして
学習します。

別のamber ablationとして、AHI/ABI/FCIに対応すると推定されるbandを公開仕様から近似カテゴリへ
写像し、配布uint8上で次を作ります。運営確認なしにgreen系へ混ぜません。

- IR windowの値、局所min/mean/std、gradient、texture
- upper/lower WV − window
- 8.5/8.7µm − window
- split-windowの差だけでなくratio/rank/local anomaly
- CO2-window関係
- 3 frameの絶対差と、整列後のcooling residual
- solar geometryによるVIS/NIR信頼度soft gate

実際のAHIは16 bands、Full Disk 10分で、可視0.5–1 km、赤外2 kmの異解像度です。
[JMA AHI specification](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html)
ただし配布TIFFのband別scale/offsetは未同定です。raw uint8上の`WV-window`やratioは物理的な
brightness-temperature differenceやradiance ratioではなく、**経験的DN spectral contrast**です。
learnable affineを後置しても物理量へ復元したことにはなりません。絶対Kelvin thresholdは移植せず、
差・比・rank・local anomalyをセンサーごとに学習します。画像ごとのmin-max normalizationは
frame間の絶対DN orderingを壊すため、controlled ablationなしには使いません。

共有encoderの上で3–5個の無名soft gateを学習し、各gateの経験的DN contrast・texture・降水分布を
事後診断します。`warm/liquid`、`ice/cirrus`、`deep convection`等の物理名は独立検証できるまで
付けません。exp032のconditional satellite headはMeteosat-only fold0で悪化しており、
hard expert一般を棄却する証拠ではありませんが、まずexpertの大半を共有し、sensor embeddingと
経験的cloud regimeだけを条件付ける保守的な設計にします。rain occurrenceとpositive amount branchは
soft cross-branch interactionで連携させます。
[MTCF](https://www.mdpi.com/2072-4292/13/12/2310)

### 5.3 「Meteosat」はFCIとして再監査する

配布名`vis_04 ... ir_133`の16 channelは、従来MSG/SEVIRIの12 channelより、MTG FCIの
16 channel構成と強く一致します。これは一次資料と列名からの**推論**であり、metadataまたは
運営回答で確定が必要です。
[EUMETSAT MTG FCI L1c guide](https://user.eumetsat.int/resources/user-guides/mtg-fci-level-1c-data-guide)

exp017のMeteosat mappingをSEVIRI前提で解釈していた箇所があれば再監査します。競技band名に
基づくカテゴリ対応と、FCI L1cのradiometry、calibration、resolution、projectionを仮定することは
分離します。後者は確認できていないため使いません。AHI/ABI/FCIは近いbandを持ってもspectral
responseが同一ではなく、外部数値SRFは運営確認まで入力に使いません。

### 5.4 P1: Motion-compensated Cloud-top Cooling

raw flowの`u,v`や予測fieldのadvected smoothingではなく、衛星3 frameから次を作ります。

1. IR windowまたはtextureを±1–2 pxの離散相関で整列する。
2. IR window、WV-window、8.5-windowの時間差を整列後に計算する。
3. 冷却、BTD変化、絶対innovation、局所growthを入力にする。
4. 相関gainが小さいpixel/sceneではconfidence gateで0へ戻す。

JMAのrapidly developing cumulonimbus/CCI処理も、雲移動を考慮した時間変化、局所min/mean/std、
複数IR差を利用します。外部AMVやJMA productは使わず、アルゴリズムの考え方だけを配布frameで
再構成します。
[JMA CCI/RDCA technical report](https://www.data.jma.go.jp/mscweb/technotes/msctechrep62-2.pdf)

### 5.5 P1: Product-aware Causal State-space Model

一般的なIMERG生成系がPMW snapshot間をmorph/Kalman/IRで埋めることを踏まえ、競技targetでも
同種のsignalがあるかを検証し、モデルを`latent precipitation + observation innovation`として
解釈する候補です。targetのrun/provenance確定前はproduct sourceを教師として直接当てません。

- 過去予測を単純平均せず、current IR innovationとmodel disagreementから連続gainを出す。
- gainはtrain OOFだけで学習する。
- row内3 frameだけのfilterはstrict green候補。過去方向だけでも別のevaluation rowを読むfilterは
  書面確認までamberにする。
- sourceを離散的に当てるmixtureは、手元EDAで二峰性がなかったため採用しない。

これは「本当の雨の物理」と「IMERG product emulator」を明示的に分ける論文化要素になります。

### 5.6 P1: Regime-conditioned Ensemble Controller

固定の衛星別weightから一歩進め、OOFだけを用いて次の小さなcontrollerを学習します。

- satellite IDと承認済みsolar geometry
- predicted tile amount / wet fraction
- model間disagreement
- temporal innovation/confidence
- zero-input / missing-frame mask

出力はstrict-row-onlyに再学習したexp016/017/018-familyのsimplex weightです。既存artifactを
そのまま使う版はsuccessor由来のamberで、patched champion由来ならredも継承します。地点IDや
eval target情報は使いません。base OOF上でfitして同じOOFで評価するとmeta-level overfitが
残るため、location単位のouter cross-fittingでcontrollerの性能を評価します。

### 5.7 P2: Train-only self-supervision

20 train locations / 28 location-month blocksしかなく時間相関も強いため、外部foundation model
より、配布train画像だけの補助taskを優先します。

- t-30,t-20,t-10からt時刻のIR/WVを予測するfuture-satellite head
- masked spectral-temporal reconstruction
- band-group dropoutとsensor-shared band-category embedding
- 1 frame → 3 frame → 5 frameのcurriculum

評価画像をpretrainingへ使うtransductive版はamberです。train-only版を先に実施します。
設計参考として[GAIA](https://arxiv.org/abs/2505.18179)を参照できますが、外部weightや学習統計は
持ち込みません。

### 5.8 P2: Regime-conditioned Constrained Displacement Correction

JAXA P-TreeのHimawari L1 gridded imageryは、地理格子化されても雲頂高度に起因するparallaxを
補正しません。ただし、これは配布TIFFの前処理が同じことを保証しません。
[JAXA P-Tree FAQ](https://www.eorc.jaxa.jp/ptree/faq.html)

そこで0 displacement初期値の小さなresampling layerを置き、方向・大きさをtrain targetから
強い制約付きで学習します。「冷たいほど大きい」は全球単調制約でなくsensor/regime-conditioned
hypothesisとして比較します。このshiftはparallaxだけでなく、wind shear、雲頂と降水の位置差、
IMERG morphing、navigation誤差も吸収し得るため、独立診断なしにparallax correctionとは呼びません。
外部の衛星直下点をhard-codeしない版はgreen寄り、公式定数を使う版は運営確認までamberです。

## 6. 論文化するための設計

### 6.1 コンペ内で主張できること

主論点は、少数地点・多センサー・量子化入力・未見地点という条件下で、

1. mean-intensity/shape factorizationがpixel hurdleよりamount calibrationを改善するか。
2. empirical spectral soft regimeがhard satellite headよりdomain generalizationを改善するか。
3. direct tile-norm trainingがMSE trainingより公式評価へ整合するか。
4. motion-compensated coolingがraw temporal stackよりconvective growthを捉えるか。

です。評価はoverallだけでなく、satellite、day/twilight/night、target amount bin、wet fraction、
unseen location、positive-only RMSE、centroid/shape errorで分解します。5-fold location splitの
平均とfold分散、locationを外側clusterとするpaired bootstrapを報告します。location内の不確実性も
見る場合だけ、その内側でtemporal-block resamplingをnestします。row単位や、locationを跨がない
temporal-blockだけのbootstrapは行わず、独立cluster自体が少ないため小さな差へ強い有意性を
主張しません。

### 6.2 コンペだけでは主張できないこと

IMERGをtargetにして「真の降水推定が改善した」とは言い切れません。IMERG自身がgeo-IRと
morphingを含むためです。コンペ後のscientific trackでは、

- input: Himawari/ABI/FCI
- primary reference target: DPR/CORRA等の独立に近い降水観測
- auxiliary target: IMERG
- evaluation: warm rain / cirrus / deep convection / limb / day-night

を設計し、同じモデルがIMERG模倣と実降水でどう順位を変えるか検証します。
DPR/CORRAも完全なground truthではなく、sampling、attenuation、retrieval、時空間collocationの
誤差を持ちます。したがって、DPR/CORRAだけでmodel selectionするholdoutを固定し、IMERG auxiliaryの
on/off ablationを行います。外部研究データはコンペ環境から完全分離した上で、JAXA/NICT、NASA、
EUMETSAT等の利用・再配布・論文掲載条件をdatasetごとに確認します。
full-spectrum GEOからCORRAを推定し、IMERGをpretrainingに使う近年の研究は参考になりますが、
コンペ中はデータもweightも使いません。
[Oya](https://arxiv.org/abs/2511.10562)

### 6.3 Paper candidate title

`Metric-Audited Area-Mean Intensity–Shape Factorization with Quantization-Robust Spectral Regime Mixtures for Cross-Sensor Geostationary Precipitation Retrieval`

中心仮説の候補はmean-intensity/shape factorization + metric auditです。ただしspatial median、
Energy Score、scale/shape分解自体は既知であり、このコンペ一つだけでpublishableとは断定しません。
`metric-aligned`にはserver定義の確定、`quantization-robust`には量子化強度を変えるcontrolled
test、`cross-sensor`にはleave-one-sensor/location-out、さらに独立target/datasetとprior-art比較が
必要です。spectral regimeとcoolingは、その検証を満たした場合の追加貢献候補です。

## 7. 締切までの実験順

公式締切は2026-07-27 23:59 GMTです。実験は次の順に固定します。

### Phase 0: 今日

1. 運営へ§3.4を質問する。
2. geocoding公式回答のpermalink、topic/message ID、投稿日を回収し、ローカル和訳と照合する。
3. `strict-green`、`amber-exploration`、`red-archive`のsubmission registryを分ける。
4. strict baselineをCSV-row-only、no patch、no cross-row smoothingで再構築する。
5. exp018 configの`encoder_weights: imagenet`等は現architectureでは未使用であることを監査記録に
   残し、将来の誤読を防ぐ。exp018本体はcustom ConvBlockでscratch学習している。

### Phase 1: 低コストで高い情報価値

1. strict source OOF field ensembleのmean / median / geometric median比較。weight学習はouter cross-fit。
2. direct tile-norm fine-tune。
3. tile mean-intensity head追加の2-fold screening。
4. two foldsとも同方向かつ概ね`-0.003`以上なら5-foldへ進める。

### Phase 2: 本命

1. tile mean intensity × normalized shapeを5-fold。
2. strict input版とamber successor版を同一codeで分ける。
3. OOFでamount bin、satellite、day-night別の改善を確認する。
4. 現在の近縁model群に対するOOF→LB回帰残差RMS `0.0041`未満の差をPublic一発で結論にしない。
   これは一般的なLB noise推定値ではなく、10件のin-sample calibration residualにすぎない。

### Phase 3: 直交性のある追加案

1. 配布値だけで作るquantization-aware empirical spectral featureのablation。
2. 勝てばsoft regime + cross-branch interaction。
3. motion-compensated cooling。
4. OOF controllerでchampion同士をblend。

### Phase 4: 最終化

1. green系をclean checkout相当の環境から再学習・再推論する。
2. manifestにGit commit、config、weight hash、data checksum、license、OOF、postprocessを保存する。
3. zip内29,090 TIFF、非負・有限値、shape、dtype、GeoTIFF metadata、CSV順序を検証する。
4. red/amber artifactが参照graphへ混入していないことを機械的に確認する。
5. 入賞時のsource提出で一発再現できるcommandを実行して記録する。

## 8. Stop rules

- model容量追加は、2-fold両方で改善しない限り5-foldへ進めない。
- exp032のconditional-head arm、exp035の統合dilated-bottleneck arm、rotation TTA、
  prediction-level advected smoothingは同一条件では再開しない。dilation単独等を一般化して棄却しない。
- diffusion/generative modelは、mean-shape champion確立後、normalized shape residualだけを対象にする。
- Public差`<0.004`は、OOF・衛星別・fold別に一貫しない限り勝敗と呼ばない。この値を普遍的な
  leaderboard noiseとは解釈しない。
- successor、evaluation cross-row filter、eval SSLは、公式許可が来なければgreen finalへ昇格させない。
- overlap patchは最終候補へ戻さない。

## 9. 主要一次資料

- [Solafune competition page](https://community.solafune.com/competitions/f87811b8-1964-4f4b-84b3-6fddd67ec4b1)
- [Solafune official competition API](https://production.server.solafune.com/api/v3/competition/f87811b8-1964-4f4b-84b3-6fddd67ec4b1)
- [Solafune metrics implementation](https://github.com/Solafune-Inc/solafune-tools/blob/main/solafune_tools/metrics.py)
- [NASA IMERG V07 technical documentation](https://gpm.nasa.gov/resources/documents/imerg-v07-technical-documentation)
- [JMA Himawari-8/9 AHI specification](https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html)
- [NOAA GOES-R ABI specification](https://goes-r.noaa.gov/spacesegment/abi.html)
- [EUMETSAT MTG FCI L1c guide](https://user.eumetsat.int/resources/user-guides/mtg-fci-level-1c-data-guide)
- [JMA CCI/RDCA technical report](https://www.data.jma.go.jp/mscweb/technotes/msctechrep62-2.pdf)
- [JAXA P-Tree FAQ](https://www.eorc.jaxa.jp/ptree/faq.html)
- [GPROF-IR](https://arxiv.org/abs/2605.07167)
- [Oya](https://arxiv.org/abs/2511.10562)
- [MTCF](https://www.mdpi.com/2072-4292/13/12/2310)
- [Gneiting and Raftery 2007](https://doi.org/10.1198/016214506000001437)
- [Chaudhuri 1996](https://doi.org/10.1080/01621459.1996.10476954)
- [Tweedie precipitation objective](https://arxiv.org/abs/2509.08369)

## 10. 最終判断

優勝に向けて最も合理的なのは、`.66617`をさらに磨くことではなく、そこで得た知見を
strict-safeな強いモデルへ移植することです。最初にmean-intensity/shape factorizationとmetric audit、
次にquantization-aware empirical spectral regimes、最後にOOF controllerを行います。

論文候補の差別化は「新しい巨大network」ではなく、**評価ノルム、target productの生成物理、
量子化された多衛星domain shift、面平均強度と形状の誤差診断を一つの推定問題として検証する
こと**です。新規性は独立dataset・controlled test・prior-art比較が揃って初めて主張します。
この順序なら、仮説が外れてもnegative resultが明確に残り、コンペの規約安全性とscientific
validityを同時に守れます。
