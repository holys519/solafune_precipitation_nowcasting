# Submission Registry — 規約リスク区分 (green / amber / red)

*作成: 2026-07-17。基準: `doc/research_survey_v3_2026-07-16.md` §3。*
*派生物は upstream の最も厳しい区分を継承する。「valid表示」は手法の承認ではない。*

## 【2026-07-20 追記2】公式裁定・完全版 (announcement post, 全項目に回答)

運営が正式アナウンスを投稿 (先の個別回答より詳細)。**推論時は「T時点で存在するデータだけで
予測する」が原則** ("as if deployed live")。

**許可**:
- timestamp ≤ T−10 minのobservationは他行由来でも可 (T−40, T−50...とさらに過去も可)
- 過去データのみから作る派生特徴 (rolling統計、データセット全体の画像統計、provided画像での
  教師なし事前学習)
- 自分自身の過去 (< T) の予測値の再利用 (自己回帰/recursive手法)
- **予測値に対する時間方向平滑化・後処理は、対象が全てT以下 (causal/過去方向) のみ許可**
- 学習時にtrainフォルダの未来フレームを教師信号として使うのは可 (推論時のリークがないため)

**禁止**:
- timestamp ≥ Tのobservationを予測Tに使用 (successor rowのT, T+10, T+20フレームを含む) — 確定
- **non-causalな平滑化** (Tの予測を、Tより後の対象時刻の予測で補正すること) — これは
  「後の予測はT以降の観測由来」という理由で明確に禁止
- 外部ソースからのeval正解値の取得・復元 (GPM IMERGアーカイブ等)
- **eval画像をtrain画像に照合してtargetを復元/近似すること (reverse engineeringに該当)** —
  overlap patch技法はこれに直接該当し、完全に禁止確定
- 検証: 勝者のコード検査で「各予測時刻Tより前のデータのみに切り詰めて再実行し、提出結果と
  一致するか」を実際に検証する。再現できない/上記に違反する解は失格。

**影響のまとめ**:
1. `context_rows: 2` (successor row) を推論入力に使う exp016/017/018/035系、およびこれらを
   起源とするblend (exp009系, exp015, exp024, exp026, exp027, exp033, exp036, exp037, exp039,
   exp042, **exp044**) は全て **red確定、最終提出不可**。現在の"総合ベスト" exp044 (0.6568) は
   除外。
2. exp036/exp037の行間平滑化 (`center_weight`/`prev_weight`/`next_weight` の双方向設計) は、
   仮にsourcesがgreenでも **平滑化自体が単独でred** (next側=non-causal)。
3. overlap patch (`exp014`, `apply_overlap.py`) は**完全に禁止確定** — 元々red運用だったため
   実害なし、判断は正しかった。
4. 学習時のみ未来フレームを補助教師信号として使うのは可、という新情報がある — 推論時の入力を
   `context_rows: 1`に保ったまま、学習時だけ補助損失で未来情報を使うアーキテクチャは理論上
   green足りうる (未着手、優先度は要検討)。
5. **新しく開けた道**: 自己回帰的に自分の過去予測を入力特徴として使うこと、および
   causal-only (pastのみ) の予測平滑化は明確に許可された — 既存のbidirectional平滑化を
   prev-onlyに変更すればgreenとして復活できる可能性がある。
6. 締切は運営のアナウンス日から1週間延長 (実質 2026-08-03頃、要最終確認)。

**今後の最終提出候補は `context_rows: 1` (exp038系・exp040系・exp045) のみ** から選定する。

## 区分の定義 (要約)

- **green**: 推論入力はtimestamp ≤ T−10のobservationのみ (他行由来も可)。予測後処理は
  causal (対象がT以下) のみ。配布trainからfitした後処理。最終提出可。
- **amber**: 外部仕様由来のband対応表 (exp038_featuresの波長整列など) — 上記2026-07-20の
  裁定では扱われていない、別系統の未確認事項。運営の書面回答が得られるまで探索専用。
- **red**: successor row入力・non-causal (未来方向) な予測平滑化・overlap patch・reverse
  engineering (2026-07-20付で全て規約違反確定)、外部データ。最終提出に使わない。
- 旧分類の「行間平滑化はamber」は撤回 — bidirectional (next側を含む) はred確定、
  causal (prevのみ) はgreenと確定した。個別に判定すること。

## 旧・区分の定義との対応 (履歴用、上記が最新)

- ~~amber: successor row入力、evaluation複数行を読む後処理 (行間平滑化など)~~ → 上記の通り
  それぞれred/red-or-greenに確定済み。

## 提出済みzipの区分

| Submission | Public RMSE | 区分 | 理由 |
| --- | ---: | --- | --- |
| exp001-exp008, exp010 各zip | 0.7250-0.7938 | green | 行内入力のみ、後処理はOOF由来 |
| exp011 | 0.7232307883574975 | green | 行内入力のみ。旧strictチャンピオン |
| **exp038** | **0.6891638997287517** | **green (現strictチャンピオン)** | current-row-only 54ch、scratch 5-fold |
| exp009 | 0.7153438899106017 | **red (2026-07-20確定)** | successor row入力 (context_rows: 2) |
| exp015 | 0.7096658388930687 | **red (2026-07-20確定)** | exp009 checkpointsを継承 |
| exp016 / exp017 / exp018 | 0.6978 / 0.6997 / 0.6929 | **red (2026-07-20確定)** | successor row入力 (context_rows: 2, 105/163ch系) |
| exp024 各blend | 0.6919-0.6961 | **red (2026-07-20確定)** | red sourcesのblend |
| exp014 | 0.6968727727408199 | red | overlap patch |
| exp026 / exp027 patched / exp033 patched | 0.6747-0.6849 | red | overlap patch + successor由来sources (二重にred) |
| exp036_per_satellite_blur1_thr0p2_patched | 0.6706858062196032 | red | patch + successor由来sources |
| exp036_per_satellite_sm0p25_blur1_thr0p2_patched | 0.6661746681900441 | red | patch + 行間平滑化(amber) + successor由来sources |
| exp037 (TTA) patched | 0.666259584999578 | red | 同上 |
| exp036_per_satellite_blur0p5_joint_patched | 0.6652621793536686 | red | patch + 行間平滑化 + successor由来sources |
| exp036_per_satellite_blur0p5_joint_raw | 0.6824222826340521 | **red (2026-07-20確定)** | patchなしだが行間平滑化(amber) + successor由来sources(red) |
| exp039_4src_joint_patched | 0.6619116739607654 | red | overlap patch + successor由来sources |
| exp039_4src_joint_raw | 0.6789588628265085 | **red (2026-07-20確定)** | patchなしだがsuccessor由来sources |
| exp042_5src_joint_raw / patched | 0.6778 / 0.6608 | **red (2026-07-20確定)** | successor由来sources (exp016/017/018/035) + patch(patched版) |
| exp044_5src_scalecorr_raw / patched | 0.6577 / **0.6568 (旧・総合ベスト)** | **red (2026-07-20確定)** | 同上。**最終提出候補から除外** |

## 未提出アーティファクトの区分

| Artifact | 区分 | 備考 |
| --- | --- | --- |
| exp033/034/036/037 の `*_raw.zip` 群 | **red (2026-07-20確定)** | successor由来sources (+行間平滑化はamberのまま) |
| exp035系 checkpoints | **red (2026-07-20確定)** | successor row入力 (context_rows: 2) |
| exp038 (config_features.yaml) | amber | 波長整列表が外部仕様由来 (successor rowとは無関係、この裁定の影響なし) |
| g_eda/exp003-005 OOFキャッシュ・スイープ | **red由来 (2026-07-20確定)** | exp016/017/018由来。green判断には流用しない (スクリーニング専用) |
| exp034 threshold zips | **red (2026-07-20確定)** | exp016/017/018 sources |
| exp043_zero_baseline (all-zero, no model) | green | 診断専用。regime-shift仮説の検証用、スコア改善目的ではない |
| exp040 / exp040_metric | green、**exp040_metric単体LB確認済み 0.6955180267195701 (2026/07/20)** | current-row-only (context_rows: 1)。単体ではexp038/exp046より劣るが、アーキテクチャ多様性を持つ2本目としてTrack G3ブレンド候補 |
| exp038_canonical_only / engineered_only / sigmafixed | green (canonical_onlyのみ後述の理由でamber寄り) | context_rows: 1。canonical_onlyは波長整列表を使うため実質exp038_featuresと同じamber理由を継承 |
| exp045 (視差/parallaxレジストレーション補正) | green、**ただしfold0/4ゲートで棄却 (両fold悪化、2026-07-21)** — 最終ブレンド不採用 | context_rows: 1。Himawariのみg_eda/exp005の閉形式幾何補正 (位置・視野角から算出、外部データ不使用) |
| exp046_causal_smoothed (exp038 + causal-only時間平滑化) | green、**LB確認済み 0.6889118106607066 (2026/07/20) — 新green champion** | 2026-07-20裁定の"causal smoothingは許可"に基づく新規post-process。next_weight=0、center=0.85/prev=0.15は旧bidirectional設定からの流用で未OOFチューニング。未チューニングでもexp038単体比-0.00025、OOFで重みを最適化すればさらなる伸びしろの可能性 |
| exp047 (local solar time / hemisphere / day-of-year特徴) | green | context_rows: 1。座標はNominatim経由で凍結取得 (`g_eda/exp005/geocoded_locations.csv`、クエリ・URL・取得日時を記録済み、discussion/approved_geocoding_sources_ja.md準拠)。生の緯度経度は入力に含めず、sin/cos位置・時刻の閉形式特徴のみ使用 (訓練地域気候の暗記を避けるため) |
| exp048 (fog/低層雲BTD: mir 3.9um − win 10.5um) | green | context_rows: 1。配布バンドのみの閉形式差分 (`doc/domain_knowledge_review_2026-07-20.md` §2.4)。g_eda/exp008のclear-sky合成でbihar/dhakaが極値パーセンタイルでも霧的なままだった知見が動機 |
| exp049 (IR窓局所テクスチャ: 3x3標準偏差) | green | context_rows: 1。配布バンドの局所統計量のみ (`doc/domain_knowledge_review_2026-07-20.md` §2.4)。対流雲 vs 霧/層状雲の判別 |
| exp050 (split-window BTD差分版: spl−win) | green | context_rows: 1。既存engineered特徴の比(SPL/(win+1))に対する、文献標準の差分版アブレーション |
| exp038_seed123 / exp038_seed456 | green | context_rows: 1、exp038と同一アーキテクチャ・特徴。Track G3グリーンブレンド用のアンサンブル多様性(seed違いのみ) |
| exp038_sigmafixed | green、**5-fold提出物完成、LB未確認** | context_rows: 1。fold0/4ゲート通過済み (両fold改善)、5-fold完了・提出パイプライン実行済み |

## 運用ルール

1. green系の学習・OOF・blend・calibrationにamber/red artifactを混入させない
   (blend重み・平滑化係数・calibration曲線の「値」の流用も不可 — green OOFで再fitする)。
2. ~~運営回答でsuccessor row入力が許可されたらamber→greenへ昇格~~ → **2026-07-20付で否決確定**。
   successor row系 (context_rows: 2) は今後もredに固定し、この表の更新対象から外す。
   行間平滑化は運営が明示的に裁定済み: causal (対象時刻が全てT以下) はgreen、non-causal
   (未来の対象時刻の予測を混ぜる) はred。exp036/exp037のbidirectional設計 (next_weight>0) は
   red確定。causal-onlyへの作り直しはgreen候補として再検討可。
3. overlap patchはいかなる回答でも最終提出へ戻さない (survey v3 §8)。
4. **最終提出は必ずgreen (またはgreen由来のamber: 波長整列表など) の中から選定する。
   red (successor row・overlap patch使用) はLBスコアがどれだけ良くても最終提出の候補にしない**
   — 運営が code inspection で照合すると明言しているため。
