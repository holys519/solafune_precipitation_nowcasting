# ディスカッション追加取得分の分析 (2026-07-19)

*関連: `discussion/discussion_all.md`(生ログ追記済み)、`doc/discussion_insights.md`、
`doc/research_survey_v3_2026-07-16.md`、`l_eda/exp003`(E-3)、`l_eda/exp004`(E-4)。*

## 1. 【最重要】評価指標の定義が2チームで真っ向対立 — 我々の実測が決着させる

- **shafimiakhil**: 「LBはpooled RMSE(全pixelを一度に集計してsqrt一回)」と主張。根拠はLightningの
  `on_epoch=True`バッチ平均が楽観的になることの発見(§1)で、これ自体は正しい指摘だが、
  「LBがpooledを採用している」という結論は**実際の提出スコアとの照合による検証ではない**。
- **Bull**: 「LBは per-image RMSEを計算してから平均(pooledではない)」と明言。実際の提出結果に
  基づく("In our validation, the LB score behaves as...")。
- **discussion_all.md既存ログ** (LB 0.69解法のコード, 2884行目): コメントに
  `"THE metric: per-sample RMSE (per tile sqrt(mean(sq err)), then plain mean over tiles)"`
  と明記 — 3人目の独立ソースがBull側を支持。

**我々の実測 (l_eda/exp003, E-3, 15ペア) は決定的にBull側を支持する**:
tile平均(per-image)予測子のSpearman **0.971**(残差std0.0044) vs 全画素pooled予測子のSpearman
**0.782**(残差std0.0137)。3ソース中2つがBullと同じ結論で、かつ我々だけが実際に多数の
提出ペアで定量検証している。**shafimiakhilの§1の中心的主張(LBの集計方式)は誤りと判断し、
採用しない。** ただし彼らの手法上の指摘(Lightningのバッチ平均が自分のローカル検証を歪める点)
自体は正しく、自分のパイプラインでバッチ平均ロギングを使っていないか確認する価値はある
(我々はanalyze_oof.py/train.pyで`tile_rmse`を毎回のバッチ平均ではなくサンプル単位のsqrtで
計算しているため、この罠には該当しない)。

## 2. E-4 (fold分散=regime構成) のクロスチーム裏付け

shafimiakhilは独自に「fold RMSEをheld-outのwetnessに回帰するとR²=0.78-0.97」を発見し、
「最良に見えたfoldは単に一番乾燥していただけ」と結論した。**これは我々のE-4
(`l_eda/exp004/FOLD_ANATOMY.md`)と完全に同じ発見**であり、2チーム独立で再現されたことで
確信度が上がる。彼らのbihar/dhaka(fold4)が「モンスーン地域なのに異常に乾燥
(bihar: mean 0.00066mm, 99.57%ゼロ, p99=0.00)」という指摘は、**我々のexp038系オラクル
ラダーやfold anatomyでも同じ2地点がfold4に入っており**、独立検証の価値がある未解決の
データ品質フラグとして記録する(§5参照)。

## 3. 重雨規模の誤差集中とdiscrepancyの解釈

shafimiakhil: heavy(≥2mm)は画素の4.13%だが誤差の85.7%(2アーキテクチャで84.2%/85.7%、
再現性あり)。Bull: 「low-intensityはほぼ無価値、誤差の大半はmid('broadly wet')帯にある」
と主張し、**heavy tailではなくmid帯への投資を推奨**。両者は矛盾するのではなく、
「heavy(≥2mm)」と「mid('broadly wet', 恐らく0.05-2mm程度)」の境界定義が違うだけの可能性が
高い。我々自身のE-1オラクルラダー(amount_swap Δ-0.06)は「量誤差が支配的」という結論では
両者と整合するが、**どのbinに投資すべきかは自分のOOFで再検証する価値がある**
(既存のregime binは`l_eda/exp004`で[0, 0.01, 0.1, 0.3, 1.0, 100]を使用しており、
Bullの言う"mid帯"とほぼ対応。r2/r3(0.1-1.0mm)がむしろ主投資対象という可能性は
まだ定量化していない)。

## 4. 較正(calibration)の限界 — 自分たちの実測と整合

shafimiakhilのisotonic再較正失敗(leave-one-fold-out: 悪化、oracle: 0.8%改善のみ)は、
**我々自身の観測(exp015のisotonicはexp009 baseで小さな改善はあったが、exp023での
mean/median比較やexp006のtrue-mean-scale分析でも「較正だけでは大きく伸びない」)と
整合的**。特に重要なのは§5aの「climatology-specificな較正はfoldを跨いで転移しない」
という発見 — これはE-4の帰結と同じ構造で、**evalの18地点がtrainと非重複である以上、
train OOFで作った較正曲線をevalへそのまま適用するリスク**を再確認させる。

## 5. データ品質: 要フォローアップ

| 項目 | 報告内容 | 我々への影響 | 対応 |
| --- | --- | --- | --- |
| GOES 4バンド破損 (peppamint) | eval `upper_midwest`(28件)・`rio_grande_do_sul`(14件)が集中、連続10分刻みで4/16バンドのみ | **両地点とも我々のeval setでGOES使用と確認済み**。`dataset.py`の`read_satellite_raw`は`n_chan=min(arr.shape[2],channels)`で汎用パディングしており、**クラッシュせず正しく処理される**(コード修正不要)。ただし該当タイルは実質12/16chの情報を欠いたまま予測している | 対応不要(既存実装で安全)。ただしupper_midwest/rio_grande_do_sulの個別tile_rmseが他地点より悪ければ、この情報欠損が原因の可能性を念頭に置く |
| IR window(win band)値=0のuint8アンダーフロー疑惑 (ccilabo) | 強雨域とClean Longwave IRのcount=0が重なる | 我々のengineered features(`frame_engineered_channels`)は`win`を`SPL/(win+1)`等で使用しており、win=0付近で異常値が出る可能性 | **未検証。次のEDAで自分のtrain/eval winバンドのゼロ率と降水量の相関を確認する価値がある**(exp017/038_features系の物理特徴の信頼性に関わる) |
| bihar/dhakaの異常な乾燥 (shafimiakhil) | 2023-2026全期間でbiharがmean 0.00066mm・p99=0.00 | 我々のfold4にもbihar/dhakaが含まれる(E-4のlocation_anatomy参照)。**追加調査済み**: `g_eda/exp008`のclear-sky合成(別コミュニティ投稿の時系列パーセンタイル手法、10/98パーセンタイルで雲除去)をbihar/dhaka(Himawari)とupper_midwest/rio_grande_do_sul(GOES)に適用したところ、後者2地点は湖・川・農地がくっきり見える一方、**bihar/dhakaは極値パーセンタイルを取っても最後までモヤがかったまま**(NDVI植生検出率0.000、水域検出率0.001-0.046、いずれも異常に低い)。両地点のファイルは12月末開始(冬季)— インド・ガンジス平野の**冬季放射霧(晴天・非対流条件で発生し、衛星からは雲のように見えるが降水を伴わない)**という仮説と整合的。単月データのみに基づく推測で確証ではないが、「見た目は曇天なのに実測降水がほぼゼロ」という一見矛盾する所見に、単なるデータ品質問題ではない気候学的説明を与える | 仮説として記録。追加の月/年のデータがあれば再検証したい |

## 6. 公式ルーリングの新規追加分

- **geocoding follow-up確定**: 標高・海岸線距離・気候区分は**外部データ扱いで禁止**
  (既存`discussion/geocoding_coordinates_ja.md`と同内容、確定表現がより明確になった)。
  許可されるのは位置・時刻の閉形式関数のみ(緯度経度・半球・三角関数エンコード・
  太陽ジオメトリ/現地太陽時)。
- **train/eval入力画像の用途は全面的に許可**(am45k質問への回答、2026-07-19以前は
  未確認だった内容): 教師なし目的・sanity check・前処理開発への利用、evalの衛星tif
  (placeholderターゲットを除く)を前処理統計や教師なし目的に使うことも明示的にYES。
  **これは`g_eda/exp003-005`のOOFキャッシュ手法や、evalの衛星入力統計を使う前処理
  (例: 衛星別normalization statsをeval込みで計算)の適法性を裏付ける。**
  ただし「placeholderターゲットからGPM-IMERGを推定する試みは不可」と明記。

## 7. Bullの実務Tips — 我々の残り期間戦略との整合性チェック

| Bullの助言 | 我々の現状 | 差分 |
| --- | --- | --- |
| ensembleの多様性は「新しい情報」、アーキテクチャの違いではない | exp035_no_dilation/exp038_featuresを加えた4/5-sourceブレンドは実際に効いている(OOF改善が確認済み) | 方針一致。継続 |
| 全データ再学習(fold分割なし)を最後に行う | 未実施 | **survey v3計画のPhase 4に「全data再学習」を明示的に追加する価値あり** |
| public-privateのshakeupは小さい見込み、0.000x差のLBは無視 | 我々のE-3ノイズ閾値(残差std 0.0041-0.0044)と整合 | 既存方針を維持 |
| 損失関数は複雑にしすぎない(plain MSE in log1p space が勝った) | 我々はhurdle log-normal/multiscale/aux-maskなど複雑な構成 | **直接の反証ではないが、Occamの剃刀として今後の実験でシンプルな対照を必ず取る価値がある** |

## 8. LBスナップショット更新 (2026-07-19時点、shafimiakhil経由)

top10のスプレッドはわずか0.0156 (1位0.6295 → 10位0.6452)。公式all-zero baselineは0.91265
(これまで我々が使っていた0.746/0.962はtrain側のローカル推定値で、公式baseline値とは
異なる可能性がある — 要確認)。我々の現amberチャンピオン(0.67778, exp042 raw)は
top1との差**0.0483**、top10との差**0.0326**。
