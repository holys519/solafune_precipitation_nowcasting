# Round 5 実験計画 — ディスカッション知見 × exp016-034 結果の統合

*作成: 2026-07-16*
*根拠: `doc/discussion_insights.md`(公式ディスカッション分析)、`outputs/analysis/exp016-034`、`doc/public_scores.md`、`doc/plan/discussion_all.md`*

---

## 0. 現在地の整理

### Public LB

| Submission | Public RMSE | 内容 |
| --- | ---: | --- |
| LB 1位 (2026-07-10時点) | 0.6422 | — |
| 「0.677の壁」(タイル平均オラクル) | 0.6766 | ディスカッション#17の実測 |
| **exp026 (現ベスト)** | **0.67465** | exp024 `equal_016_017` blend + exp014 overlap patch |
| exp024 `equal_016_017` (patch前) | 0.6919 | exp016+exp017 50/50 |
| exp014 | 0.69687 | exp009 + overlap patch |

overlap patch込みでちょうど壁を割った位置。トップとの差 ~0.032 で、その主成分は依然として
**タイル内位置特定**(壁の下のwet/dryマスクrung = 0.083のヘッドルーム)。

### 5-fold OOF tile_rmse(モデル単体)

| Exp | OOF tile_rmse | 差分(対exp009) | 内容 |
| --- | ---: | ---: | --- |
| exp009 | 0.6239 | — | successor-row 105ch |
| exp016 | 0.6186 | −0.0053 | hurdle log-normal head |
| exp017 | 0.6163 | −0.0077 | 物理チャネル+波長整列 163ch |
| **exp018** | **0.6093** | **−0.0147** | 高解像内部処理+aux wet-mask+multi-scale MSE |

exp018(G-032、局在化3機構)が単体ベスト。ディスカッションの「壁の下は位置特定」という
診断と整合して、**局在化投資が最も効いた**。

### exp028-032 fold0 ablation(対照: exp017 fold0 = tile 0.30642 / positive 1.96848)

| Exp | 内容 | fold0 tile_rmse | fold0 positive_rmse | 判定 |
| --- | --- | ---: | ---: | --- |
| exp028 | target-time-first + \|Δ\|チャネル (165ch) | 0.30120 (−0.0052) | 1.96887 (±0) | **採用候補** |
| exp029 | 衛星別anchor IR rain proxy | 0.30482 (−0.0016) | 2.00727 (+0.039) | **棄却**(採用条件のpositive RMSE非悪化を満たさない) |
| exp030 | dilated bottleneck (d=2,4) | 学習中best 0.29994 (−0.0065) | 未確定 | **最有力だが要再解析**(下記P-1) |
| exp031 | Focal+Tversky補助loss | epoch 1以降 valid NaN | — | **棄却→原因修正が先**(下記P-2) |
| exp032 | 衛星別head分岐 | 0.30882 (+0.0024) | 1.94669 (−0.022) | **棄却**(全体悪化。positive改善は下記E-2の材料) |

### exp018 OOFの局在化診断(wet tile 32,230枚)

| 衛星 | spatial corr | wet IoU@0.25 | centroid距離(px) |
| --- | ---: | ---: | ---: |
| goes | 0.482 | 0.319 | 4.97 |
| himawari | 0.425 | 0.319 | 6.13 |
| **meteosat** | **0.307** | **0.186** | **8.14** |
| 全体 | 0.397 | 0.269 | 6.54 |

**Meteosatの位置特定だけが著しく悪い**。テスト構成はhim 39% / met 39% / goes 22%なので、
Meteosat局在化の改善はLB寄与が大きい。

### 既知の検証上の問題

- fold0 (france/friuli) はtarget_mean 0.104と全体(0.289)より遥かに「楽」。fold間tile_rmseは
  0.29〜0.81と3倍近く振れる。**fold0単独のA/Bは採択判断として弱い**。
- OOF全体(0.609)とpublic生値(0.692)のギャップ ~0.08。eval地点はtrainと非重複なので当然だが、
  このギャップの**構造**(どのtrain foldがevalに近いか)を未計測。CVとLBの単調性は
  提出済み15点で検証可能なのに未実施。

---

## 1. プロセス修正(実験より先に片付ける)

### P-1. exp030の解析やり直し

`outputs/analysis/exp030/analysis_summary.json` は `folds: 0`・tile_rmse 0.3986 と、
学習ログのbest(0.29994 @epoch51)と乖離しており、**checkpoint保存前またはモデル定義不一致の
状態でanalyze_oofが走った**とみられる。ジョブ3933461/3933474完了後に
`g_model/exp030/best_model_fold0.pt` に対して解析だけ再実行し、fold_summary.csvを再生成する。
dilated bottleneckは fold0 で最大の単独改善候補なので、これを直さないとRound 5の合成判断
(exp035)ができない。

### P-2. exp031のNaN原因特定

fold0はepoch 1でvalid NaN化、fold1-4も1-2 epochで停止。焦点は
(a) Tversky分母のゼロ割り(εなし)、(b) focalの`log(p)`でp=0、(c) AMP下のoverflow。
`losses.py`にε付与・`torch.clamp(p, 1e-6, 1-1e-6)`・loss項のfloat32計算を入れ、
weight 0.05→0.01のwarmupで再走。**ただし優先度は低**(G-021系。局在化本命が先)。

### P-3. ドキュメント同期

- `EXPERIMENT_PLAN.md` にexp028-034の行を追加(現状exp027まで)。
- `doc/public_scores.md` が2026-07-10で停止。exp016/017/018/024/026/027の提出結果と
  exp026=0.67465を反映。**CV-LB対応表(下記E-3)の入力データになるので正確に**。
- exp033/exp034がuntracked。コミットする。

---

## 2. 追加EDA(意思決定に必要な情報の取得)

いずれも学習不要・既存OOF/予測から計算できるものを優先。置き場所は `l_eda/exp003`
(CSV+PNG+レポートの既存パターンに従う)。

### E-1. 予測のオラクル分解ラダー【最優先】

**問い**: 残误差のうち「量の誤り」と「配置の誤り」はそれぞれ何点分か。壁の下0.083の
ヘッドルームのうち、我々は今いくら取れているか。

**方法**: exp016/017/018のOOF予測とGPM truthから、ディスカッション#17のラダーを
**我々の予測側**で再計算する:

1. `flat(pred)`: 予測をタイル平均で均した場合のtile_rmse(=我々の「量」情報のみの得点)
2. `flat(truth)`: 正解タイル平均(壁 0.6766 の再現確認)
3. `truth_amount × pred_mask`: 正解の量 × 我々のwetマスク(マスク品質の寄与)
4. `pred_amount × truth_mask`: 我々の量 × 正解マスク(量品質の寄与)
5. `blur(pred, σ=1,2,3)`: 予測をぼかした場合(過剰シャープネスの罰の有無)

**判断**: (1)−(実際のOOF) が「配置で稼げている量」。(3)(4)の比較で次の投資先
(マスク改善 vs 量改善)を数値で決める。5.のblurが改善するなら、後処理ガウシアンぼかし
または学習時blur-tolerant lossが即効の勝ち筋(σ=2px化した正解が0.38を出す世界なので、
展開が過剰にシャープなら罰されている)。衛星別に必ず分解する。

### E-2. 変位の系統性分析(Meteosat局在化の原因切り分け)

**問い**: centroid距離8.1pxのMeteosat誤差は**系統的な変位**(パララックス/registration不足)か、
**ランダムな配置失敗**(情報不足)か。

**方法**: exp018 OOFのwet tileごとに pred/truth の雨域重心変位ベクトル(dy, dx)を計算し、
(location, satellite)別に平均と分散を出す。`outputs/l_eda/exp001/parallax_shift_by_location.csv`
の既存推定と突き合わせる。

**判断**:
- 平均変位が有意(例: |mean| > 2px)で方向が揃う → **exp013a(G-017)のregistration前処理を
  再起動**(実装済み・A/B未実施)。学習不要でdataset.pyのflag一つ。
- 分散のみ大 → registrationでは直らない。Meteosat入力の情報量問題(E-6へ)。

### E-3. CV→LB回帰の校正【安価で価値が高い】

**問い**: OOF tile_rmse(あるいは衛星構成重み付きOOF)はLBの順序をどこまで予測するか。
「LBデルタ<0.005はノイズ」の閾値は我々のパイプラインでも正しいか。

**方法**: 提出済みペア(exp001-027で public RMSE 既知の~15点)について、OOF tile_rmse・
衛星構成重み付きOOF(him .39/met .39/goes .22)・fold0のみ・fold0+4平均の4種の予測子で
LBとの順位相関と回帰残差を出す。

**判断**: 以後の採択判断に使うCV指標を**この結果で固定**する。fold0単独A/Bの
継続可否(exp028-032方式)もここで判定する。

### E-4. fold1/fold3の失敗解剖

**問い**: fold1 (0.75) / fold3 (0.79) の悪さはheavy-rain regime起因か、地点固有か。

**方法**: `oof_sample_metrics.csv`(exp018)を地点×target_mean分位で層別し、
tile_rmse・positive_rmse・wet IoUを比較。dhaka/aceh/central_vietnam(モンスーン域)と
france/friuli の regime 差を定量化。

**判断**: heavy-rain regimeが支配的なら、intensityヘッドのtail(exp016のσ扱い、
log-normal serving)の改善やheavy-rain重点サンプリングを Round 5 後半の候補に格上げ。
LBのeval地点構成がどちらのregimeに近いかをE-3と接続して評価する。

### E-5. eval側 t+0 フレーム可用性の実測検証

**問い**: ディスカッション主張「t+0以降フレームがeval 99.4%で存在」は我々のデータでも
正確か。successor rowのタイムスタンプギャップ分布は。

**方法**: evaluation CSVの隣接行から、各行に対する successor 観測の存在率とΔt分布を
衛星別に集計(exp009のdataset.pyのロジックを流用、学習不要)。

**判断**: exp028(target-time-first)を5-foldへ進める前提確認。eval側で欠損が多い衛星が
あれば、maskチャネルのfallback品質がLBを律速する。

### E-6. Meteosat固有の情報量監査

**問い**: Meteosatの局在化が悪いのはセンサー(チャネル対応・量子化・zero率88.5%)由来か、
アフリカ/欧州の降水regime由来か。

**方法**: `l_eda/exp002`のbt_rain_response・band_healthをMeteosatに絞って再実行し、
波長対応表(`doc/discussion_insights.md` §3表)のMeteosat列をband統計で再検証。
IR窓(idx13)とrainの相関を、同regime(target_mean層別)でhimawariと比較。

**判断**: 同regimeでも相関が大きく劣る → センサー情報限界。Meteosatだけ
per-satellite fine-tune(exp032の否定結果はhead分岐のみ。encoder込みのfine-tuneは未検証)
またはMeteosat専用後処理(強めのvalue_threshold)を試す根拠になる。

---

## 3. 追加実験(exp035〜)

番号はexp034まで使用済みのため exp035 から。すべて `g_experiments/expNNN` 標準構成
(config.yaml + run_variant.sh ハーネス)に従う。

### exp035: 勝ち筋の統合 — exp018 × exp028 × exp030【本命・最優先】

**根拠**: 3つは独立軸で各々単独有効:
- exp018: 出力側の局在化(highres内部処理+aux mask+multi-scale MSE)、5-fold OOF −0.0070 対exp017
- exp028: 入力側の時間設計(target-time-first+|Δ|)、fold0 −0.0052 対exp017
- exp030: ネットワーク中央の受容野(dilated bottleneck)、fold0 −0.0065 対exp017(P-1確定後)

**実装**: exp018のmodel/lossをベースに、dataset.pyへexp028の
`features.target_time_first` / `temporal_abs` を移植(in_channels再計算、startup検証あり)。
bottleneckをexp030のdilation 2,4ブロックへ置換。batch sizeはexp030に合わせ96に落とし、
highres経路のメモリ増を考慮してOOM時はbase_channels据え置きでaccumulation 2。

**手順**(E-3の結論が出るまでの暫定プロトコル):
1. fold0 **と fold4** の2-fold A/B(fold0だけの「楽さ」対策。fold4はandalusia/bihar/
   florida/jakarta/kinshasaでregime多様)。対照はexp018の同fold(0.29234 / 0.58531)。
2. 両foldで非悪化かつ片方で>0.003改善なら5-fold本走+submission。
3. 3要素同時が悪化した場合に備え、config flagで exp030要素だけ / exp028要素だけ の
   fallback armを用意(1 armずつ切れる構成にする)。

**採用条件**: 5-fold OOF tile_rmse < 0.606(exp018比 −0.003以上)、
かつ衛星構成重み付きOOFでも改善。

### exp036: Meteosat局在化レスキュー

**根拠**: E-2/E-6の結果待ちだが、どちらに転んでも打ち手があるよう両armを準備:
- arm A(E-2が系統変位を示した場合): exp013a registration前処理をexp035の入力に接続。
  per-(location, satellite)の median shift を学習前に適用。
- arm B(ランダム誤差の場合): Meteosat行のみ損失重み1.3〜1.5(テスト39%構成対応)
  + Meteosat行に限定したaux wet-maskのDice重み増。encoder共有のままなのでexp032の
  失敗(完全head分岐)とは異なる介入。

**採用条件**: Meteosat別OOF wet IoU 0.186→0.22以上、かつ全体OOF非悪化。
片衛星だけの改善で全体悪化なら棄却(exp032の教訓)。

### exp037: blur-tolerant出力(E-1の結果次第で発動)

**根拠**: 「σ=2pxぼかし正解が0.38」= 計量は変位に鈍感な滑らかな場を好む。Moran's I 0.57、
雨画素の98.3%は隣接雨画素あり — salt-and-pepper は保証された損。

**実装**(いずれも小粒、E-1のblurラダーで効く方だけ):
- 後処理: 予測へのgaussian blur(σ∈{0.5, 1, 1.5})をOOFでsweep(学習不要、exp034型)。
- 学習時: multi-scale MSEの重み再配分(2x/4x pooled項を強める)、またはblurred-MSE項追加。
- 連結成分クリーニング: 面積<3pxの孤立wet成分を0化(OOF sweepで面積閾値を決める)。

**採用条件**: OOFで改善した設定のみ提出枠1つで検証。<0.005のLB差はノイズ扱い(E-3で更新)。

### exp038: 量子化スナップ+衛星別value_threshold(後処理小勝ち回収)

**根拠**: GPM値の99.6%が0.01倍数(l_eda/exp002 G-025、未実装)。またvalue_thresholdは
現在全衛星一律0.10だが、Meteosatはzero率88.5%で最適点が異なるはず(exp018 OOFの
value_threshold sweepを衛星別に再集計するだけで決められる)。

**実装**: exp034型の学習不要ジョブ。(1) 0.01グリッドへのスナップ、(2) 衛星別
value_threshold、(3) 両方、をOOFで確認し最良のみzip化。

**採用条件**: OOF改善が出た組だけ提出。期待値は各+0.001〜0.003の微益だが確実に正。

### exp039: Harvest(旧exp020 / G-034の実行)

**時期**: exp035-037の勝者確定後、**終盤に一度だけ**。

**内容**(計画済み仕様の再確認+更新):
- 勝者config × 3 seeds × 5 folds(exp025のシード分散結果から、seed間ばらつきは
  fold間より小さいがゼロではない — 3 seedで十分)
- TTA拡張: 現行flip 3-way → rot90/180/270込み6-way
- OOF重み付きブレンド(等重みではなく。exp024/027の等重みブレンドを置換)
- ブレンドOOFに対するisotonic再fit(exp015機構) → 衛星別に分けてfitする点が新規
- exp038の後処理 → 最後にoverlap patch(規約確認は下記)

### 継続判断が必要な既存実験

- **exp029(IR rain proxy)**: fold0でpositive RMSE悪化により採用条件未達。README記載の
  「経験曲線の第二段階(trainから学習したbin曲線)」へ進む価値はE-6の結果次第。
  デフォルトでは**クローズ**。
- **exp031**: P-2の修正後、fold0のみ再走。それでもexp017比フラットなら**クローズ**
  (占有GPU時間に対する期待値が低い)。
- **exp032**: クローズ。ただしfold0でpositive_rmse −0.022の部分効果があったので、
  exp036 arm Bの損失重み設計の参考にする。
- **exp033/exp034(提出運用)**: 既存のadaptive submission orderに従い実施。結果を
  `doc/public_scores.md`へ記録し、**E-3の回帰に追加**(patch有無・threshold有無の対照が
  4点増えるのはCV校正上も貴重)。

---

## 4. 検証プロトコルの変更(Round 5から適用)

1. **報告指標**: OOF tile_rmse に加え、**衛星構成重み付きOOF**(him .39 / met .39 / goes .22)
   と**衛星別OOF**を`analysis_summary.json`に常設(analyze_oofへ追加。E-3で予測力を検証後、
   採択判断の主指標を決定)。
2. **単fold A/Bの廃止**: 採択判断は最低 fold0+fold4 の2-fold。fold0のみは「実装検証」
   (学習が回る・NaNが出ない)にのみ使う。
3. **局在化指標の常設**: spatial_correlation / wet_iou_025 / centroid_distance_pixels は
   exp018で導入済み。**全実験のanalyze_oofに含め、採用条件に明記**する(tile_rmse同点なら
   wet IoUが高い方を採る)。
4. **LBノイズ閾値**: E-3完了までは従来通り0.005。完了後に更新。

---

## 5. 提出計画(4枠/日想定)

| 日 | 枠 | 内容 |
| --- | --- | --- |
| Day 1 | 1-2 | exp034 `thr_w018_000` → exp033 `w018_050`(既存adaptive planの分岐判定) |
| Day 1 | 3-4 | 分岐表に従いexp033/034の残り |
| Day 2+ | — | exp035の5-fold勝者(単体+patch)、exp038後処理、以後はexp039 harvestに温存 |

**規約ノート(重要)**: overlap patch(exp014系)は「重複するtrainタイルのGPM正解を評価タイルへ
コピー」する手法。exp033 READMEに記載の通り、**公式に許可が確認できない限り最終提出に
使わない**。全ブレンドで `--skip-patch --zip-raw` のraw版を並行生成し、規約安全トラックを
常に維持する。最終2枠は「patch版ベスト」と「raw版ベスト」の両建てを推奨。

---

## 6. 優先順位まとめ

| 優先 | 項目 | 種別 | 工数 | 期待値 |
| --- | --- | --- | --- | --- |
| 1 | P-1 exp030再解析 + P-3 ドキュメント同期 | プロセス | 小 | 判断基盤 |
| 2 | E-1 オラクル分解ラダー | EDA | 小(OOF既存) | 投資先の確定 |
| 3 | E-3 CV→LB回帰校正 | EDA | 小 | 以後全採択判断の信頼性 |
| 4 | exp035 統合モデル | 学習 | 大(2-fold→5-fold) | −0.005〜−0.010 OOF |
| 5 | E-2 変位系統性 + E-6 Meteosat監査 | EDA | 中 | exp036の分岐決定 |
| 6 | exp038 量子化スナップ+衛星別threshold | 後処理 | 小 | +0.001〜0.003×複数 |
| 7 | exp036 Meteosatレスキュー | 学習 | 中 | met 39%構成でレバレッジ大 |
| 8 | exp037 blur-tolerant(E-1次第) | 後処理/学習 | 小 | E-1が正なら即効 |
| 9 | exp039 Harvest | 学習+統合 | 大 | 終盤一括+0.005前後 |
| 10 | P-2 exp031修正、E-4/E-5 | 混合 | 小 | 保険・確認 |

---

## 7. 実行状況 (2026-07-16 追記)

| 項目 | 状態 | 結果 / 備考 |
| --- | --- | --- |
| P-1 exp030再解析 | **完了** (job 3934665, 3933474完了後に依存実行) | fold0確定: tile 0.29994 / positive 1.9188 — 両指標でexp017比改善、採用候補確定。**`g_model/exp030`のfold1-4 checkpointは1-2 epochの残骸**で、5-fold OOF 0.6534は無効 (fold_summaryはfold0のみ有効) |
| E-3 CV→LB回帰 | **完了** (`l_eda/exp003`, 9ペア) | 5-fold OOF: Spearman **0.90**・残差std **0.0033** / **fold0単独: Spearman 0.50でほぼ無情報** / fold0+4: 0.78。§4のプロトコル変更 (fold0単独A/B廃止、LBノイズ閾値~0.005) を定量的に確認。衛星重み付きOOFは素のOOFと同等 |
| exp035 実装 | **完了+スモーク合格** (job 3934664) | `g_experiments/exp035`: exp018コード + exp017 dataset (features) + drop_zero + dilated bottleneck。3アーム (full 165ch / no_dilation / dilation_only 105ch) のforward/backward+実データパイプライン検証済み。次: `sbatch singularity_run.sh config.yaml 0` と `... 4` |
| E-1 オラクル分解ラダー | **完了** (job 3934675, `outputs/g_eda/exp002`) | 下記「E-1の結果」参照 — 優先順位を変える発見 |
| P-3 ドキュメント同期 | **完了** | `EXPERIMENT_PLAN.md` にexp028-035行+ログ追記、`doc/public_scores.md` にexp026 (0.67465) / exp024 (~0.6919) 反映。exp024の正確な桁とexp026の提出時刻はSolafune提出リストから要補完 |
| E-3の含意 | — | exp028/exp030の「採用候補」はfold0のみの弱い証拠。exp035の最終採否は必ず5-fold OOF (残差std 0.0033基準) で判断する |
| P-2 exp031修正 | **完了・再投入** (job 3934743) | NaN原因特定: AMP下でrain_probがfp16のため `clamp(1e-5, 1-1e-5)` の上限がfp16で1.0に丸められ、dry画素で `log(1-prob)=log(0)=-inf`。`exp017/losses.py` のFocalTverskyRainLossをfp32計算に修正 |

## 8. E-1の結果 (2026-07-16) — 優先順位の更新

exp016/017/018すべてで同じ構造 (global tile_rmse, `outputs/g_eda/exp002/*_oracle_ladder.json`):

| 反実仮想 | exp018 | 読み |
| --- | ---: | --- |
| actual | 0.6093 | 実スコア。**既にtrain OOFでは壁(0.6766)の下** |
| amount_swap (量だけ正解、配置は我々) | **0.5446** | **+0.065の伸び代 — 支配項** |
| mask_swap (配置オラクル、量は我々) | 0.7111 | actualより悪化 — 我々の空間パターンは既にフラットマスクより良い |
| blur_pred_s1 (σ=1ぼかし) | 0.6070 | 一貫して −0.002 の無料改善 (σ=2はほぼ中立) |
| flat_pred (我々の量のみ) | 0.7010 | 配置情報は+0.09分の価値を既に生んでいる |

**結論と計画修正**:

1. 我々の3モデルの残差は**配置ではなくタイル量の誤差が支配的**。ディスカッションの
   「壁の下は位置特定」はexp018以前の話で、exp018の局在化投資が効いた後の今は量が律速。
2. → **exp035に `config_tilemean.yaml` アームを追加**(タイル平均MSE項 `tile_mean_weight: 0.3`。
   multi-scaleラダーの最粗段を直接教師)。exp036 (Meteosat局在化) は優先度を下げる。
3. exp023の結果 (global isotonic/linear calibrationはOOF改善なし) とも整合: 量誤差は
   タイルごとに符号が違うので**グローバルな値域補正では直らない** — 学習で直す必要がある。
4. blur σ=1は小さいが3モデル一貫の無料改善 → exp037のOOF sweep (σ 0.5〜1.5) は
   提出前後処理として実施する価値あり。ただしE-3のLBノイズ閾値(~0.005)未満なので
   単独提出はせず、他の改善に同乗させる。
5. train OOFでは壁の下 (0.609) なのにLB生値は0.69 — **eval地点への汎化ギャップ(~0.08)が
   もう一つの構造問題**。E-4 (fold失敗解剖) とE-6の優先度は維持。

## 9. 提出リスト全量の反映 (2026-07-16, ユーザー提供)

新規判明分 (`doc/public_scores.md` 全面更新済み):

| Submission | Public RMSE | 含意 |
| --- | ---: | --- |
| exp018単体 | 0.6929495140301676 | **単体ベスト**。exp017比LB −0.0068 vs OOF −0.0070 — E-3回帰の実例通り |
| exp016 / exp017 単体 | 0.6978 / 0.6997 | OOF序列 (017<016) がLBで逆転。Δ0.002はノイズ帯 — 両者は同格扱い |
| exp024 equal_016_017 | 0.6919274860606568 | 正確な桁を確定 (概算0.69193は正しかった) |
| exp027 patched 2種 | 0.6807 / 0.6849 | **exp026 (0.6746) より悪化** — seed checkpointの等重みブレンドは希釈。exp039はOOF重み+弱メンバー除外が必須 |
| exp009入りblend | 0.6961 / 0.6940 | OOFの劣るモデルを混ぜると悪化 — ブレンド構成はOOF準拠 |

E-3再校正 (12ペア): 5-fold OOF Spearman **0.951** (modelのみ0.964)、残差std 0.0041。
fold0単独は0.79へ持ち直したが序列は不変。ノイズ閾値は~0.004-0.005で確定。

**次の提出候補の優先順** (4枠/日):
1. `exp034_thr_w018_000_patched.zip` (threshold効果の単離、既存プラン通り)
2. `exp033_w018_050_patched.zip` (exp018ブレンド効果 — exp018単体0.6929がequal_016_017の
   0.6919に肉薄しており、混合で改善する見込みが高い)
3. 以降はexp033/034 READMEのadaptive分岐表に従う
4. exp035の5-fold完走後は、その提出とexp033型ブレンドの再構成を優先
