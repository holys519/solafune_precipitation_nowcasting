# Public Scores

Last updated: 2026-08-01

**2026-08-01 追記11: exp064_convnext_lr2e4(ConvNeXt-Tiny)を発掘。2026-07-27にfold0/4「タイ」
判定のまま放置され、5-fold学習済みなのに一度もOOF集計・LB提出されていなかった。5-fold OOFは
0.60442(effv2sの0.60443とほぼ同値)で、最下位だったconvnext_small(LB 0.67934)より明確に
良い可能性が高い — 「同系統内は小さい方が勝つ」パターンの追加確認候補。zip構築済み、
LB未提出、明日の提出優先候補の一つ。**

**2026-08-01 追記10: exp065 6-wayアンサンブルが新champion。「アーキテクチャの異質性は単体で
勝てなくてもブレンドに効く」という仮説がLBで実証された。**

- `exp065_champion_ensemble_6way_causal.zip` = **0.6670330551889574** (2026/08/01 20:14:05)。
  v2 (0.6684780268241325) 比 **-0.00145**、v1 (0.6696622672831182) 比 **-0.00263**で**新
  champion**。pvt_v2_b0とswin_smallを4-wayブレンドに追加した6-way版。honest nested OOFの改善は
  v1比-0.00395(0.58813→0.58418)で、実測LB改善(-0.00263 vs v1)への転移率は約67% — このプロ
  ジェクトの低自由度変更にしては標準的だが、単体では勝てなかったpvt_v2_b0/swin_smallを
  含めたこと自体が主要因である点が新しい。`nested_blend.py`のouter-cross-fit重みでpvt_v2_b0が
  最大(0.2487)を取った — swin_lr2e4が2026-07-30に示した「最も異質なメンバーが最大重みを得る」
  パターンが再現し、**「単体LBで勝てない新backbone系統でも、既存メンバーと十分異質ならブレンド
  多様性として実際にLBを押し上げる」**という仮説が2例目の実測で裏付けられた。effb3_seed456は
  同アーキで多様性が低いため未採用(2-seedアンサンブル`exp064_effb3_seed_ens_42_456_submission.zip`
  は別途構築済み、LB未提出)。

**2026-08-01 追記8: pvt_v2_b0/effb3_seed456/swin_smallを実測、3件ともchampion更新なし。
「pretrained backbone内の容量スケールアップ」軸はEfficientNet/ConvNeXt/Swinの3系統全てで
EXHAUSTED確定。新backbone系統の微小OOF差はLB順位を予測しない、pretrained系統でもseedノイズは
再現する、の2点も判明。**

- `exp064_pvt_v2_b0_lr2e4_submission.zip` = **0.672541575619076** (2026/08/01 11:49:04)。新backbone
  系統(PVTv2)、5-fold OOFは今セッション測定した単体モデル中**最良**(0.59617、effb3の0.59758を
  上回る)だったが、実測LBはexp064_effb3(0.6720097338314985)比 **+0.00053とノイズ帯内の実質タイ**。
  OOFの微小な優位(-0.0014)はLB順位に反映されなかった — 「明確なOOF差(0.005+)は転移するが、
  0.001台の差は転移しない/ノイズに埋もれる」という解像度の下限を示す新知見。championは更新
  されず、新backbone系統への乗り換えによる単体性能向上はここで頭打ち。
- `exp064_effb3_seed456_submission.zip` = **0.6758692035492916** (2026/08/01 11:49:30)。effb3の
  seed456版。OOFはseed42比+0.00932悪化していたが、実測LB悪化は+0.00386(exp064_effb3比)で
  OOF差の約41%のみ転移。exp056で確認済みのseedノイズ(~0.007)が**pretrained backbone系統でも
  同様に再現**することを確認。単体championにはならないが、exp056の2-seedアンサンブル
  (seed42+456が両メンバーを上回った前例)と同じ手法をeffb3にも適用する価値がある — 次候補。
- `exp064_swin_small_lr2e4_submission.zip` = **0.6719466133572829** (2026/08/01 12:05:44、
  再提出で確定 — 初回は internal server error で未採点)。OOFではswin_lr2e4(Tiny)比-0.00080の
  改善だったが、実測LBはswin_lr2e4(0.6704384663890527)比 **+0.00151の悪化**。effb4/convnext_small
  と同じ「OOF改善→LB悪化」のパターンで、**「pretrained backbone内での容量スケールアップ」軸は
  EfficientNet・ConvNeXt・Swinの3系統全てでEXHAUSTED確定**(唯一の期待株だったSwinも例外では
  なかった)。この系統内スケールアップというレバーはもう残っていない。

**2026-08-01 追記9: effb3 2-seedアンサンブルzip、および6-way championアンサンブル候補zipを構築。
どちらもLB未提出(提出枠待ち)。6-wayは`submission_gate.py`でクリーンなGO判定、v1のnested OOF
(0.58813)を大きく更新する最終OOF 0.58418を確認。**

- `outputs/submissions/exp064_effb3_seed_ens_42_456_submission.zip` — effb3 seed42+456の等重み
  平均(`g_experiments/exp064/build_seed_ensemble.py`、exp056方式を移植)。exp056の2-seed
  アンサンブルが両メンバーを上回った前例を踏まえた低リスク候補。OOF検証はまだ、LB未提出。
- `outputs/submissions/exp065_champion_ensemble_6way_causal.zip` — exp065championを
  exp056/effb3/effv2s/swin_lr2e4の4-wayから、pvt_v2_b0とswin_smallを加えた**6-way**へ拡張。
  `g_eda/exp011/nested_blend.py`のouter-cross-fit重み(pvt_v2_b0=0.2487/effb3=0.2035/
  swin_lr2e4=0.1938/swin_small=0.1615/exp056=0.1425/effv2s=0.05)+この6-way自体のOOFに対して
  再チューニングしたcausal smoothing(`g_eda/exp010`、blur_sigma=0→0、himawari/meteosat
  閾値0.12)。**nested(honest outer-cross-fit)score 0.58493 (best solo pvt_v2_b0の0.59787比
  -0.01294、overfitting_gap -0.00052)、後処理込みの最終OOFは0.58418** — v1championのnested
  OOF (0.58813) を大幅に更新。`l_eda/exp005/submission_gate.py`は**GO**(point delta -0.01264、
  80%CI `[-0.01567,-0.00982]`がゼロ除外、P(better)=1.000、gainは17/20 locationに分散し
  上位3location集中は36.3%でgeography shortcut兆候なし)。LB未提出、次の提出枠での最優先候補。
  effb3_seed456はこの6-wayには含めていない(effb3と同アーキで多様性が低く、exp056の
  seed_ensemble系の教訓通り希釈リスクがあるため、2-seedアンサンブルとして別トラックで評価)。

**2026-07-31 追記7: exp065 v2(causal smoothing再チューニング)が新champion。「同系統内スケール
アップ」軸は完全にEXHAUSTED。exp066(temporal architecture)は早期に負け筋と判明、gate段階で打ち切り。**

- `exp065_champion_ensemble_v2_causal.zip` = **0.6684780268241325** (2026/07/30 23:08:29)。
  v1 (0.6696622672831182) 比 **-0.00118**。変更点はブレンド重みではなく後処理(causal smoothingの
  tap重み・blur sigma・衛星別閾値)の再チューニングのみ — 低自由度の変更はOOF→LB転写が安定して
  効くという、このプロジェクトの一貫した経験則を改めて裏付けた。**新champion。**
- `exp064_effb4_submission.zip` = **0.6771412784573648** (2026/07/31 08:30:47)。同系統の
  `exp064_effb3` (0.67201) より **+0.00513悪化**。EfficientNet系統内でのb3→b4スケールアップは
  効かない、むしろ悪化する。
- `exp064_convnext_small_lr2e4_submission.zip` = **0.6793384249392268** (2026/07/31 08:31:17)。
  全候補中最下位。tiny版(fold0/4 tie)と合わせ、ConvNeXt系統はサイズによらず本タスクとの相性が
  悪いと判断。
- → **「pretrained backbone内でのcapacityスケールアップ」軸はEfficientNet/ConvNeXt両系統で
  EXHAUSTED**。唯一fold0/4をクリーンPASSしたSwin系統でのみ、swin_small(exp064_swin_small_lr2e4)
  で追試中(2026-07-31投入、結果待ち)。
- `exp056_wd5x`/`exp056_wd10x`(weight_decayスイープ、fold0/4のみ): 両方とも基準
  (fold0 0.28159/fold4 0.58503)に対し悪化 (wd5x: +0.00447/+0.00281、wd10x: +0.00204/+0.00048)。
  **正則化(weight_decay)強化は効かない、EXHAUSTED。**
- `exp066`(temporal architecture、ConvLSTM/attention fusion by shared per-frame encoder、
  exp063の crude channel-stack history を置き換える設計、context_rows:2で exp063_cr2 と同一情報量):
  `exp066_convlstm_cr2`のfold0=0.28713/fold1=0.72759(best_epoch=1で20epoch改善なし=最適化不安定)、
  fold0はcr1 baseline (0.28159) にも exp063_cr2 (0.28302、crude stackで既に悪化・close済み) にも
  劣る。exp063と同じfrom-scratch系統はfold0/4ゲートに偽陰性の前例がない(偽陰性が確認されている
  のはpretrained backbone軸のみ)ため、ゲートを信頼してconvlstm_cr2の残り3fold・submitと
  attention_cr2の残り3fold・submitを打ち切り(fold0/1のみ完走)。**「crude stackの失敗はアーキ
  テクチャの問題ではなく、履歴情報自体がこの情報量では効かない」という、より強い形でexp063の
  結論が補強された。** pretrained encoder×temporal fusionのハイブリッドなど、より有望な派生形は
  残り日数の制約で見送り。

**2026-07-30 追記6: exp065アーキテクチャ多様アンサンブルが新champion。Track Aの手法設計が実証された。**
`exp065_champion_ensemble_causal_submission.zip`(実ファイル名`exp065_champion_ensemble_causal.zip`)
= **0.6696622672831182** (2026/07/30 16:03:44)。exp064_effb3(0.67201)比 **-0.00235** — 今
キャンペーン最良スコア。同時刻に提出された`exp064_swin_lr2e4_submission.zip`単体も
**0.6704384663890527** (2026/07/30 16:04:15、effb3比 -0.00157) で旧championを上回った。

**手法の検証結果(`g_eda/exp011/nested_blend.py`、2026-07-30新規実装のN=4 outer-cross-fit)**:
- メンバー: `exp056`(from-scratch)/`exp064_effb3`/`exp064_effv2s`/`exp064_swin_lr2e4`(transformer)
- honest nested OOF score **0.58813** vs best solo (effb3) **0.60004** → **-0.01191**
- in-sample score 0.58764、overfitting_gap **-0.00049**(ほぼゼロ = in-sampleとnestedがほぼ一致、
  exp055のときのような過学習インフレは今回は起きていない)
- `l_eda/exp005/submission_gate.py`: **GO** (point delta -0.00944、80% CI
  `[-0.01270, -0.00611]`がゼロを除外、gainは地域に分散＝ジオグラフィshortcutの兆候なし)
- 採用重み: effb3=0.315 / **swin_lr2e4=0.315**(同率最大 — 最もアーキテクチャが異質な
  transformerメンバーが最大の重みを得た。多様性仮説の裏付け) / exp056=0.27 / effv2s=0.10(最小
  — effb3と同系統エンコーダで多様性価値が低いことと整合)
- OOF改善(-0.01191)のLB転移率は約20%(-0.00235/-0.01191)。過去のブレンド事例(~47-51%)より
  低いが、方向は完全に一致し、gate GO・実測プラスの健全な結果

**fold0/4ゲートの信頼性についての追加知見**: `exp064_swin_lr2e4`はfold0/4ゲートで明確な
**PASS**(f0 0.27404改善/f4ノイズ内)だった数少ない候補で、実際にLBでも旧champion超え
(0.67044)を達成した。一方effb3/effv2s/convnext_lr2e4は「mixed/tie」判定だったが実際は
大きく勝った(effb3)/引き分けだった(convnext)。→ **ゲートが完全に信頼できないのではなく、
「明確なPASS」は依然シグナルとして機能し、問題は「mixed/tie」判定を安易に「効果なし」と
解釈してしまうこと**、と結論を精緻化。

**2026-07-30 追記5: exp064_effb3が大幅更新の新green champion。想定と食い違う重大な結果。**
`exp064_effb3_submission.zip` = **0.6720097338314985** (2026/07/30 07:16:38)。旧champion
`exp056_seed_ens_42_456` (0.68277) 比 **-0.01076** — このプロジェクトのノイズ帯(~0.004-0.005)を
はるかに超える改善。`exp064_effv2s_submission.zip` = **0.6740019855472996** (2026/07/30 07:17:24)
も旧champion比 **-0.00877** で2番手。

**これはround8計画のfold0/4ゲート判定と真っ向から矛盾する:** ゲート時点ではeffv2sは
fold0 0.28208/fold4 0.58531で champion (0.28159/0.58503) と実質タイ、effb3は
fold4 0.58135(good)/fold0 0.28833(worse)の「mixed」判定で、どちらも「capacity ≠ bottleneck」
という結論の根拠になっていた。しかし実際の5-fold LBでは両方が旧championを圧倒し、特にeffb3の
勝ち幅はexp056が単体からのアーキテクチャ変更で得た改善(-0.00268)の4倍。**fold0/4の2-fold
ゲートは、pretrained backboneがもたらす真の汎化改善を捉えられていなかった可能性が高い** —
訓練/評価地域の非重複という本コンペの核心的困難に対し、pretrainedのcapacity/汎化力は
これまでの診断(「capacityは効かない」)より遥かに有効だったことになる。**"Pretrained backbone
capacity ≠ bottleneck" というEXHAUSTED判定は撤回**。次の一手はeffb3を中心にした追加backbone
探索・アンサンブル(champion 2-seed × effb3など)を最優先で検討すべき。

**2026-07-26 追記4: seed-ensembleが新green champion。** `exp056_seed_ens_42_456`
(exp056 seed42 と seed456 のeval予測の等重み平均) = **0.6827721114076882**、exp056単体(0.68396)を
**-0.00119** 更新。ポイント: (1) この2シード平均は**両メンバー(seed42=0.68396, seed456=0.68569)の
どちらより良い** — 誤差相殺による本物のアンサンブル効果。(2) 3シード平均(42+123+456)=0.68408 は
seed123(単体0.69111の外れ値)に引かれseed42とほぼタイ → **弱いメンバーを入れると希釈する**
(exp055の教訓のseed版)。(3) **seedだけでLBが~0.007ばらつく** (seed42 0.68396 / seed456 0.68569 /
seed123 0.69111) と判明、かつOOF順位(42<123<456)とLB順位(42<456<123)が不一致 —
我々の多くの試行(±0.002級)がseedノイズ以下だった。追加seed(789/1337/2024)を学習中で、
champion級seedのみで大きめのアンサンブルを組む。**注意: 42+456の2本選択は公開LBによる
seed選択(mild public-LB overfit)なので、追加seed完走後は全champion級seedの平均をprivate向けの
頑健な最終候補にする**。

**2026-07-24 追記3: exp056が新green champion (0.68396、-0.00268)。** Mean-intensity×
normalized-shapeの分解アーキテクチャ(`g_experiments/exp056`、`g_eda/exp002`のoracle-ladder
分析が示した「残差の支配項はplacementでなくAMOUNT」という知見に基づく、strict-green・
context_rows:1、feature/blend側の微調整ではなく正面からのアーキテクチャ変更)。
`l_eda/exp005/submission_gate.py`でも事前にGO判定(OOF Δ-0.00576、80%CI
`[-0.00920,-0.00259]`で0を除外、18/20 locationで改善・集中なし)が出ており、実測でも
方向が一致した(OOF改善の約47%がLBに転移 — exp050/047のような逆転ではなく、
exp040_metric単体のような過小transferでもない、健全な範囲)。**Green championを
exp038_sigmafixed (0.68664) からexp056 (0.68396) に更新。**

**【2026-07-20 公式裁定・完全版】** 運営の公式アナウンス投稿で全項目が確定した。詳細は
`doc/submission_registry.md`。要点:
- successor row入力 (context_rows: 2) 禁止確定 — T以降のobservationは一切不可
- 予測後処理の時間方向平滑化は **causal (対象が全てT以下) のみ許可**、non-causal (未来の対象
  時刻の予測を混ぜる) は禁止 — exp036/exp037のbidirectional設計はこれ単独でもred
- overlap patchによるeval復元は reverse engineering として完全禁止確定
- 自己回帰的な自分の過去予測の再利用、causal-onlyの平滑化は明確に許可 (新しい green の道)
- 学習時のみ未来フレームを補助教師信号に使うのは可 (推論入力がcausalなら問題なし)
- 勝者はコード検査で「T時点までのデータに切り詰めて再実行→提出結果と一致するか」を検証される
- deadlineは運営アナウンス日から1週間延長 (実質2026-08-03頃、要最終確認)

**このtableの多くの上位スコアはsuccessor row由来のred分類となり最終提出には使えない** —
rank 1つずつに `[ELIGIBLE]` / `[RED]` を付記した。**2026-07-22現在のgreen champion:
exp038_sigmafixed (0.68664)、更新されず**。exp046_causal_smoothed (0.68891) → exp038_sigmafixed
(0.68664、-0.00227) の順で更新後、**exp055 (exp038_sigmafixed×exp040_metricのOOF最適ブレンド、
Track G3第1弾) を実測したがOOFの−0.00870改善はLBに一切transferせず、むしろ僅かに悪化した**
(0.68721〜0.68726、詳細は下記Submission LogとObservations)。exp040_metric単体 (0.69552) は
exp038単体より劣るが、Track G3ブレンド用のアーキテクチャ多様性の2本目としての価値は
ブレンド重み再設計後に再評価する。

**2026-07-24 追記**: sigma_mode=fixed (championを作った変更) がexp038 strict由来の他特徴量にも
効くか検証。exp050_sigmafixed (split-window BTD + sigma_fix) はpooled OOF 0.60643
(champion比-0.00185) と改善したが、**実測LBは0.68702 (champion比+0.00038、ノイズ帯内)** — OOFの
改善はほぼtransferしなかった。exp047_sigmafixed (solar time特徴 + sigma_fix) はpooled OOF 0.60747
(champion比-0.00081) だったが、**実測LBは0.70014 (champion比+0.01350) という致命的な悪化** —
このプロジェクトのノイズ帯(~0.004-0.005)を遥かに超える、今セッション最大のOOF/LB逆転。
原因分析は下記Observations参照。**green championはexp038_sigmafixed (0.68664) のまま、更新なし**。

**2026-07-24 追記2 (I-002: 検証手法の堅牢化)**: exp050/exp047/exp055の3件連続でOOF改善が
LBに転移しなかったことを受け、`l_eda/exp005`(location-cluster bootstrap CI +
gain-concentration監査 + submission gate)と`g_eda/exp011/nested_blend.py`(outer-cross-fit
ブレンド重み最適化)を新規実装。遡及検証では、exp050_sigmafixed・exp047_sigmafixedのどちらも
提出前に走らせていれば **NO-GO** と判定されており(train locationがわずか20個・foldが
不均等なため80% bootstrap CIが0をまたぎ、かつ改善が上位2locationに69-74%集中)、実際に
起きた「transferしない」「致命的に悪化する」という結果と整合する。

**ただしexp055は捕捉できなかった** — `nested_blend.py`でexp038_sigmafixed×exp040_metricの
outer-cross-fit再検証を行ったところ、nested score (0.60007) は元のin-sample fit (0.59982) と
ほぼ同一(overfitting gap -0.00024、fold間の重みも0.45-0.53で安定)で、「in-sample fitが
fold構造にoverfitしていた」という当初の仮説は支持されなかった。この結果を`submission_gate.py`に
通すとGO判定(80%CIが0を除外、改善は15/20 locationに分散、集中なし)になるが、**実測は
solo champion比+0.00057〜+0.00062の悪化**。つまりこの失敗はfold構成ノイズでもgeography
shortcutでもなく、**train 20地点内でのresamplingでは原理的に検出不可能なtrain-eval分布シフト**
が原因と判断される。教訓として「ブレンド候補は各構成要素自身の過去のOOF→LB転移効率
(`l_eda/exp003`の回帰: LB≈1.268×OOF−0.080)を事前チェックする」というルールを追加した
(exp040_metricは単体のOOF→LB gapがexp038_sigmafixedより大きく、これが予兆だった)。
詳細は`doc/plan/round7_validation_hardening_2026-07-24.md`、ticketは`doc/task_tickets.md`の
I-002。**今後、OOF改善が0.01未満の候補は`submission_gate.py`のGO判定なしに提出しない。
ブレンドはGO判定に加えて各構成要素の過去のOOF→LB転移効率も必ず確認する**運用とする。

This file tracks public/valid leaderboard scores for the Solafune precipitation nowcasting
competition. Metric is RMSE, so lower is better.

Sources:

- `doc/exp001_retrospective.md`
- `doc/research_survey.md`
- Solafune submission list copied by the user on 2026-07-08 / 07-09 / 07-10 / **07-16 (full list)**

## Current Best

| Rank | Experiment | Submission | Public RMSE | Submitted at | Status | Notes |
| ---: | --- | --- | ---: | --- | --- | --- |

| 1 | exp044 | `exp044_5src_scalecorr_patched.zip` | 0.6568062148127412 | 2026/07/19 11:27:19 | valid | **[RED — 2026-07-20確定, 最終提出不可]** successor-row sources (exp016/017/018/035) + overlap patch. OOF predicted only -0.00069 but realized -0.00399 vs exp042 (578% transfer, unexplained) |
| 1 | exp042 | `exp042_5src_joint_patched.zip` | 0.6607936278488564 | 2026/07/19 10:10:12 | valid | **[RED]** superseded by exp044; successor-row sources + overlap patch |
| 1b | exp039 | `exp039_4src_joint_patched.zip` | 0.6619116739607654 | 2026/07/17 11:58:12 | valid | **[RED]** successor-row sources + overlap patch |
| 2 | exp036 | `exp036_per_satellite_blur0p5_joint_patched.zip` | 0.6652621793536686 | 2026/07/17 10:27:20 | valid | **[RED]** successor-row sources + row smoothing + patch |
| 3 | exp036 | `exp036_per_satellite_sm0p25_blur1_thr0p2_patched.zip` | 0.6661746681900441 | 2026/07/16 07:33:53 | valid | **[RED]** successor-row sources + row smoothing + patch |
| 4 | exp037 | `exp037_per_satellite_sm0p25_blur1_thr0p2_patched.zip` | 0.666259584999578 | 2026/07/16 08:05:21 | valid | **[RED]** rot90 TTA tied; successor-row sources + patch |
| 5 | exp036 | `exp036_per_satellite_blur1_thr0p2_patched.zip` | 0.6706858062196032 | 2026/07/16 06:48:24 | valid | **[RED]** successor-row sources + patch |
| 6 | exp033 | `exp033_w018_050_patched.zip` | 0.671989922822016 | 2026/07/16 10:41:11 | valid | **[RED]** successor-row sources + patch |
| 7 | exp026 | `exp026_submission.zip` | 0.6746506841387548 | 2026/07/13 12:01:55 | valid | **[RED]** successor-row sources + overlap patch |
| 8 | exp042 | `exp042_5src_joint_raw.zip` | 0.6777841449591795 | 2026/07/19 09:40:47 | valid | **[RED — 2026-07-20確定]** successor-row sources (no patch, but still red on its own) |
| 8b | exp039 | `exp039_4src_joint_raw.zip` | 0.6789588628265085 | 2026/07/18 12:14:31 | valid | **[RED — 2026-07-20確定]** successor-row sources |
| 9 | exp027 | `exp027_half016_half017family_patched.zip` | 0.6806568162687938 | 2026/07/13 12:02:46 | valid | **[RED]** successor-row sources + patch |
| 10 | exp036 | `exp036_per_satellite_blur0p5_joint_raw.zip` | 0.6824222826340521 | 2026/07/17 10:27:45 | valid | **[RED — 2026-07-20確定]** successor-row sources + row smoothing |
| 11 | exp036 | `exp036_per_satellite_sm0p25_blur1_thr0p2_raw.zip` | 0.6834922402930078 | 2026/07/17 10:34:09 | valid | **[RED — 2026-07-20確定]** successor-row sources + row smoothing |
| -2 | exp065_champion_ensemble_6way_causal | `exp065_champion_ensemble_6way_causal.zip` | 0.6670330551889574 | 2026/08/01 20:14:05 | valid | **[ELIGIBLE — current green champion]** 6-way ensemble (exp056/effb3/effv2s/swin_lr2e4/pvt_v2_b0/swin_small) with weights + causal smoothing both re-fit via `g_eda/exp011/nested_blend.py`'s outer-cross-fit for this exact member set. Beats v2 by -0.00145, v1 by -0.00263. Honest nested OOF improved -0.00395 vs v1 (0.58813 -> 0.58418), ~67% OOF-to-LB transfer. pvt_v2_b0 (a solo tie with effb3) got the largest blend weight (0.2487) -- confirms architectural diversity adds ensemble value even without a solo win |
| -1.5 | exp065_champion_ensemble_v2_causal | `exp065_champion_ensemble_v2_causal.zip` | 0.6684780268241325 | 2026/07/30 23:08:29 | valid | **[ELIGIBLE — superseded by 6-way]** same 4-way ensemble as v1, only the post-processing (causal smoothing tap weights, blur sigma, per-satellite thresholds) re-tuned. Beats v1 by -0.00118 -- a low-degree-of-freedom change, transferred cleanly OOF-to-LB as usual for this category of change |
| -1 | exp065_champion_ensemble_causal | `exp065_champion_ensemble_causal.zip` | 0.6696622672831182 | 2026/07/30 16:03:44 | valid | **[ELIGIBLE — superseded by v2]** architecture-diverse 4-way ensemble (exp056/effb3/effv2s/swin_lr2e4) weighted by `g_eda/exp011/nested_blend.py`'s outer-cross-fit fit (effb3=0.315, swin_lr2e4=0.315, exp056=0.27, effv2s=0.10) + causal-only smoothing. Nested OOF -0.01191 vs best solo, overfitting_gap -0.00049 (honest), submission_gate.py GO. Beats exp064_effb3 by -0.00235 |
| -0.5 | exp064_swin_lr2e4 | `exp064_swin_lr2e4_submission.zip` | 0.6704384663890527 | 2026/07/30 16:04:15 | valid | **[ELIGIBLE — 2nd best solo]** pretrained Swin-Tiny (transformer) backbone, lr=2e-4, 5-fold. This was the one exp064 arm with a CLEAN fold0/4 gate PASS (not mixed/tie) -- and it also delivered on LB, beating exp064_effb3 by -0.00157. Refines this session's gate-reliability finding: clean passes remain trustworthy signal, the failure mode was specifically dismissing mixed/tie verdicts |
| -0.2 | exp064_swin_small_lr2e4 | `exp064_swin_small_lr2e4_submission.zip` | 0.6719466133572829 | 2026/08/01 12:05:44 | valid | **[ELIGIBLE, but a regression vs swin_lr2e4]** pretrained Swin-Small (scaled up from swin_lr2e4/Tiny), 5-fold. OOF improved -0.00080 vs Tiny, but realized LB is **+0.00151 worse** than Tiny -- same OOF-improves/LB-worsens inversion seen for effb4 and convnext_small. Closes the capacity-scale-up axis for all 3 pretrained families tested (EfficientNet/ConvNeXt/Swin) |
| 0 | exp064_effb3 | `exp064_effb3_submission.zip` | 0.6720097338314985 | 2026/07/30 07:16:38 | valid | **[ELIGIBLE — superseded by exp065 ensemble]** solo pretrained EfficientNet-B3 backbone, 5-fold. fold0/4 gate had called this "mixed" (fold4 good/fold0 worse vs champion baseline) but full 5-fold LB beat the prior champion by -0.01076, far outside noise band -- fold0/4 gate did not predict this; capacity/pretrained-backbone hypothesis reopened |
| 0a | exp064_pvt_v2_b0_lr2e4 | `exp064_pvt_v2_b0_lr2e4_submission.zip` | 0.672541575619076 | 2026/08/01 11:49:04 | valid | **[ELIGIBLE, effective tie with effb3]** solo pretrained PVTv2-B0 (new backbone family) backbone, 5-fold. 5-fold OOF was the best of any solo model measured this session (0.59617 vs effb3's 0.59758) but realized LB is +0.00053 vs effb3 -- within noise band, not a real win. Sub-0.005 OOF gaps between competitive pretrained backbones do not reliably predict LB order |
| 0b | exp064_effv2s | `exp064_effv2s_submission.zip` | 0.6740019855472996 | 2026/07/30 07:17:24 | valid | **[ELIGIBLE]** solo pretrained EfficientNetV2-S backbone, 5-fold. fold0/4 gate called this a tie with champion baseline; full 5-fold LB beats prior champion by -0.00877 |
| 0b1 | exp064_effb3_seed456 | `exp064_effb3_seed456_submission.zip` | 0.6758692035492916 | 2026/08/01 11:49:30 | valid | **[ELIGIBLE, worse than effb3 seed42]** effb3 architecture, seed456 instead of the champion seed42. OOF was seed42 +0.00932 worse; realized LB is +0.00386 worse (~41% OOF-to-LB transfer). Confirms exp056-style seed noise (~0.004-0.01) reproduces in the pretrained-backbone family too. Not a solo candidate, but a 2-seed effb3 ensemble (42+456) is now worth testing, mirroring exp056_seed_ens_42_456's success |
| 0c | exp064_effb4 | `exp064_effb4_submission.zip` | 0.6771412784573648 | 2026/07/31 08:30:47 | valid | **[ELIGIBLE, but a regression]** solo pretrained EfficientNet-B4 (scaled up from champion effb3), 5-fold. +0.00513 vs effb3 -- within-family capacity scale-up does NOT help for EfficientNet, closes that axis |
| 0d | exp064_convnext_small_lr2e4 | `exp064_convnext_small_lr2e4_submission.zip` | 0.6793384249392268 | 2026/07/31 08:31:17 | valid | **[ELIGIBLE, worst of the exp064 family]** solo pretrained ConvNeXt-Small, lr=2e-4, 5-fold. Worse than every other exp064 arm including effb4; combined with convnext_lr2e4 (tiny)'s earlier fold0/4 tie, ConvNeXt looks size-independently weak for this task, not just under-scaled |
| 11a | exp056_seed_ens_42_456 | `exp056_seed_ens_42_456_submission.zip` | 0.6827721114076882 | 2026/07/26 12:11:07 | valid | **[ELIGIBLE — superseded by exp064_effb3]** equal-weight average of exp056 seed42 + seed456 eval predictions (same green architecture, seed-only diversity). Beats exp056 seed42 solo by -0.00119 and beats BOTH members (0.68396/0.68569) -- genuine variance-reduction ensemble. Seed pairing chosen on public LB (mild overfit); to be re-based on all champion-level seeds once 789/1337/2024 finish |
| 11a2 | exp056_seed_ensemble | `exp056_seed_ensemble_submission.zip` | 0.6840805442401546 | 2026/07/26 12:10:43 | valid | **[ELIGIBLE]** 3-seed average (42+123+456); +0.00012 vs seed42 solo -- seed123 (0.69111 solo) diluted it back to ~tie. Confirms weak members hurt the average |
| 11a3 | exp056_seed_ens_4best | `exp056_seed_ens_4best_submission.zip` | 0.6846399729749617 | 2026/07/27 10:14:15 | valid | **[ELIGIBLE, worse than champion]** 4-seed average (42+456+789+2024); +0.00187 vs ens_42_456 champion -- scale-up does not help |
| 11a4 | exp056_seed_ens_6 | `exp056_seed_ens_6_submission.zip` | 0.6848948764433148 | 2026/07/27 10:14:30 | valid | **[ELIGIBLE, worse than champion]** 6-seed average (all: 42+123+456+789+1337+2024); +0.00212 vs ens_42_456 champion -- seed123 dilutes again; confirms 2-seed ensemble is the plateau |
| 11b | exp056 | `exp056_submission.zip` | 0.6839627937847801 | 2026/07/24 17:58:34 | valid | **[ELIGIBLE — best single model; superseded as champion by the 2-seed ensemble]** Mean-intensity x normalized-shape factorized architecture (`g_experiments/exp056`), strict-green context_rows:1, no blend/patch. Beats exp038_sigmafixed by -0.00268 -- an architectural change, not a feature/blend tweak. `l_eda/exp005/submission_gate.py` returned GO ahead of this submission (OOF Δ-0.00576, 80% CI excludes zero, gain diffuse across 18/20 locations) |
| 12 | exp027 | `exp027_equal_all_patched.zip` | 0.6849224439171961 | 2026/07/13 12:02:26 | valid | **[RED]** successor-row sources + patch |
| 13 | exp035 | `(recorded in E-3 audit)` | 0.6860146267326392 | — | valid | **[RED — 2026-07-20確定]** context_rows: 2 |
| 14 | exp038_sigmafixed | `exp038_sigmafixed_submission.zip` | 0.6866381028699935 | 2026/07/21 08:01:29 | valid | **[ELIGIBLE — superseded by exp056]** exp038 strict + sigma_mode=fixed (no predicted-sigma head). Beats exp046 by -0.00227 and exp038 solo by -0.00253; both fold0/4 improved during gating |
| 15 | exp055 | `exp055_global_blend_causal.zip` | 0.6872098507591518 | 2026/07/22 01:02:31 | valid | **[ELIGIBLE, but worse than solo champion]** OOF-optimal 48/52 blend of exp038_sigmafixed × exp040_metric + exp010's causal-only smoothing/blur/threshold (tuned for exp038_sigmafixed solo, not re-tuned for the blend). OOF predicted 0.59984 (−0.00868 vs exp038_sigmafixed solo's 0.60852); realized **+0.00057 worse** than solo. Major OOF/LB inversion — see Observations |
| 16 | exp055 | `exp055_global_blend.zip` | 0.6872601993829903 | 2026/07/22 01:02:58 | valid | **[ELIGIBLE, but worse than solo champion]** same 48/52 blend, no causal smoothing. OOF predicted 0.59982 (−0.00870 vs solo); realized **+0.00062 worse** than solo. Confirms the inversion is in the blend weights themselves, not the smoothing (which was a ~0 net effect in OOF too) |
| 17 | exp046 | `exp046_causal_smoothed_submission.zip` | 0.6889118106607066 | 2026/07/20 01:31:56 | valid | **[ELIGIBLE]** exp038 + causal-only temporal smoothing (center=0.85/prev=0.15, next=0, untuned). Beat exp038 solo by -0.00025; superseded by exp038_sigmafixed |
| 18 | exp038 | `exp038_submission.zip` | 0.6891638997287517 | 2026/07/18 06:01:30 | valid | **[ELIGIBLE]** strict current-row-only green model, context_rows: 1; superseded by exp038_sigmafixed |
| 19 | exp024 | `exp024_equal_016_017.zip` | 0.6919274860606568 | 2026/07/12 05:13:36 | valid | **[RED — 2026-07-20確定]** exp016/017 blend, both context_rows: 2 |
| 20 | exp040_metric | `exp040_metric_submission.zip` | 0.6955180267195701 | 2026/07/20 01:32:36 | valid | **[ELIGIBLE]** standalone green model, architecturally distinct from exp038 (metric_weight=0.6 tile-RMSE-shaped loss). Weaker solo than exp038/exp046/exp038_sigmafixed, but intended as the 2nd model for Track G3 green-blend diversity, not a solo champion |
| 21 | exp050_sigmafixed | `exp050_sigmafixed_submission.zip` | 0.6870176972903309 | 2026/07/24 16:42:13 | valid | **[ELIGIBLE, not a champion update]** split-window BTD feature + sigma_mode=fixed. OOF was champion's best (0.60643, -0.00185) but realized only +0.00038 vs champion -- within noise, essentially no transfer |
| 22 | exp047_sigmafixed | `exp047_sigmafixed_submission.zip` | 0.7001420620597125 | 2026/07/24 16:41:52 | valid | **[ELIGIBLE, but a severe regression]** solar time/hemisphere/day-of-year feature + sigma_mode=fixed. OOF looked good (0.60747, -0.00081) but realized **+0.01350 vs champion**, far outside this project's noise band. Diagnosis: not a code bug (solar_features.py's math checked out) -- likely a geography shortcut, where `hemisphere` (binary) combined with the satellite one-hot (himawari~Asia-Pacific/goes~Americas/meteosat~Europe-Africa) lets the model memorize per-(satellite,hemisphere) train-region climate baselines rather than the intended solar-time physics, exactly the risk the config's own description warned about. Withdrawn from champion consideration; do not trust this feature's OOF numbers going forward |

The complete chronological history is in `Submission Log` below.

## Submission Log

| Submitted at | Experiment | Submission | Public RMSE | User/Team | Status | Memo |
| --- | --- | --- | ---: | --- | --- | --- |
| 2026/07/07 04:29:43 | exp001 | `exp001_submission.zip` | 0.7531995875751526 | holyholyholy | valid | ローカル環境 |
| 2026/07/07 11:20:13 | exp001 | `exp001_submission.zip` | 0.7937729717031525 | holyholyholy | valid | A100テスト (reference) |
| 2026/07/08 02:25:26 | exp004 | `exp004_submission.zip` | 0.7252533726905589 | holyholyholy | valid | Two-Head Rain Detection + Amount Regression |
| 2026/07/08 02:36:58 | exp005 | `exp005_submission.zip` | 0.7445524878914139 | holyholyholy | valid | Temporal fusion |
| 2026/07/08 02:48:44 | exp006 | `exp006_submission.zip` | 0.7450324204392412 | holyholyholy | valid | Satellite-specific adapter |
| 2026/07/08 07:14:08 | exp002 | `exp002_submission.zip` | 0.7479569114058262 | holyholyholy | valid | A100_exp002 |
| 2026/07/08 09:40:50 | exp003 | `exp003_submission.zip` | 0.7522576632294679 | holyholyholy | valid | A100_exp003 |
| 2026/07/09 01:20:50 | exp007 | `exp007_submission.zip` | 0.7362157342148196 | holyholyholy | valid | Multi-exp equal-weight ensemble |
| 2026/07/09 12:36:08 | exp008 | `exp008_submission.zip` | 0.7250185237499447 | holyholyholy | valid | Official Metric + Drizzle Post-Processing |
| 2026/07/09 12:38:39 | exp009 | `exp009_submission.zip` | 0.7153438899106017 | holyholyholy | valid | Successor-Row Frames |
| 2026/07/09 12:40:03 | exp010 | `exp010_submission.zip` | 0.7348731115909746 | holyholyholy | valid | Data Cleanup Two-Head |
| 2026/07/09 12:44:58 | exp011 | `exp011_submission.zip` | 0.7232307883574975 | holyholyholy | valid | Satellite Adapter Two-Head |
| 2026/07/10 08:47:38 | exp015 | `exp015_submission.zip` | 0.7096658388930687 | holyholyholy | valid | Isotonic OOF Calibration on exp009 checkpoints (G-027a) |
| 2026/07/10 12:44:13 | exp014 | `exp014_submission.zip` | 0.6968727727408199 | holyholyholy | valid | Tile-Overlap GPM Copy Patch (post-processing on exp009 base, G-022) |
| 2026/07/11 11:23:41 | exp016 | `exp016_submission.zip` | 0.6977629323809645 | holyholyholy | valid | Hurdle log-normal head (G-030) |
| 2026/07/11 11:24:23 | exp017 | `exp017_submission.zip` | 0.6997414980565597 | holyholyholy | valid | Physics channels + wavelength alignment (G-031) |
| 2026/07/12 05:10:40 | exp024 | `exp024_blend_20_40_40.zip` | 0.693975964307325 | holyholyholy | valid | 20/40/40 exp009/016/017 blend |
| 2026/07/12 05:11:35 | exp024 | `exp024_equal_009_016_017.zip` | 0.6961199095679 | holyholyholy | valid | Equal exp009/016/017 blend |
| 2026/07/12 05:13:36 | exp024 | `exp024_equal_016_017.zip` | 0.6919274860606568 | holyholyholy | valid | exp016/017 50/50 blend |
| 2026/07/13 12:01:55 | exp026 | `exp026_submission.zip` | 0.6746506841387548 | holyholyholy | valid | exp024 equal_016_017 + exp014 overlap patch |
| 2026/07/13 12:02:26 | exp027 | `exp027_equal_all_patched.zip` | 0.6849224439171961 | holyholyholy | valid | Equal 5-way seed-family blend + patch |
| 2026/07/13 12:02:46 | exp027 | `exp027_half016_half017family_patched.zip` | 0.6806568162687938 | holyholyholy | valid | 50/50-type seed-family blend + patch |
| 2026/07/16 10:41:11 | exp033 | `exp033_w018_050_patched.zip` | 0.671989922822016 | holyholyholy | valid | 50/50 equal_016_017 × exp018 blend + patch |
| 2026/07/16 12:57:23 | exp018 | `exp018_submission.zip` | 0.6929495140301676 | holyholyholy | valid | High-res localization (G-032), best single model |
| 2026/07/16 06:48:24 | exp036 | `exp036_per_satellite_blur1_thr0p2_patched.zip` | 0.6706858062196032 | holyholyholy | valid | OOF per-satellite blend + blur + threshold + patch; OOF予測−0.0032に対し実測−0.0013 |
| 2026/07/16 07:33:53 | exp036 | `exp036_per_satellite_sm0p25_blur1_thr0p2_patched.zip` | 0.6661746681900441 | holyholyholy | valid | + temporal smoothing (0.25/0.30/0.45) (**current best**, ~rank 24) |
| 2026/07/17 10:34:09 | exp036 | `exp036_per_satellite_sm0p25_blur1_thr0p2_raw.zip` | 0.6834922402930078 | holyholyholy | valid | 3-tap smoothing stack, no patch — patch value on this stack = 0.6834922402930078 - 0.6661746681900441 = 0.0173175721029637 |
| 2026/07/17 11:58:12 | exp039 | `exp039_4src_joint_patched.zip` | 0.6619116739607654 | holyholyholy | valid | 4-source blend (+exp035_no_dilation, per-satellite weights) + joint postprocess + patch (**current best**). OOF predicted delta -0.00340 vs realized -0.00335 vs previous best — ~99% transfer, the highest-fidelity post-processing step measured so far |
| 2026/07/18 12:14:31 | exp039 | `exp039_4src_joint_raw.zip` | 0.6789588628265085 | holyholyholy | valid | 4-source blend, no patch (amber champion). Patch value = 0.6789588628265085 - 0.6619116739607654 = 0.0170471888657431 — third consistent measurement (~0.017 across 3-way ladder, 3-way joint, 4-way joint) |
| 2026/07/19 09:34:26 | exp038 | `exp038_features_submission.zip` | 0.6920702884151865 | holyholyholy | valid | current-row + wavelength-aligned physics (amber), standalone 5-fold. **OOF said this beats exp038 strict (fold0 0.28860 vs 0.28954, fold4 0.59336 vs 0.59607) but LB is WORSE than strict (0.69207 vs 0.68916, +0.0029)** — an OOF/LB inversion for the amber feature arm specifically; external-spec-derived band mapping may not generalize as well as it screens |
| 2026/07/19 09:40:47 | exp042 | `exp042_5src_joint_raw.zip` | 0.6777841449591795 | holyholyholy | valid | 5-source blend (+exp038_features) + joint postprocess, no patch (**new amber champion**). vs exp039 4-source raw: -0.00117 realized (OOF predicted -0.0023, ~51% transfer) |
| 2026/07/19 10:10:12 | exp042 | `exp042_5src_joint_patched.zip` | 0.6607936278488564 | holyholyholy | valid | 5-source blend + patch (**current overall best**). vs exp039 patched: -0.00112. Patch value on this blend = 0.0169905171 — 4th consistent measurement (~0.017 across 3-way ladder/joint, 4-way joint, 5-way joint) |
| 2026/07/17 10:27:20 | exp036 | `exp036_per_satellite_blur0p5_joint_patched.zip` | 0.6652621793536686 | holyholyholy | valid | 5-tap ±60min smoothing (per-satellite) + blur 0.5 + per-satellite thresholds + patch (**current best**) |
| 2026/07/17 10:27:45 | exp036 | `exp036_per_satellite_blur0p5_joint_raw.zip` | 0.6824222826340521 | holyholyholy | valid | same stack, no patch (amber track) — patch contribution = 0.6824222826340521 - 0.6652621793536686 = 0.0171601032803835 |
| 2026/07/16 08:05:21 | exp037 | `exp037_per_satellite_sm0p25_blur1_thr0p2_patched.zip` | 0.666259584999578 | holyholyholy | valid | rot90 TTA A/B: +0.00008の完全なタイ → TTA無効と判定、exp037クローズ |
| 2026/07/18 06:01:30 | exp038 | `exp038_submission.zip` | 0.6891638997287517 | holyholyholy | valid | strict current-row-only green model。exp011 strict比 −0.03407、strictチャンピオン更新 |
| 2026/07/20 01:31:56 | exp046 | `exp046_causal_smoothed_submission.zip` | 0.6889118106607066 | holyholyholy | valid | exp038 + causal-only時間平滑化 (2026-07-20裁定で許可、center=0.85/prev=0.15/next=0、未チューニング)。exp038単体比 -0.00025で**新green champion** |
| 2026/07/20 01:32:36 | exp040_metric | `exp040_metric_submission.zip` | 0.6955180267195701 | holyholyholy | valid | 単体green model (metric_weight=0.6のtile-RMSE整形損失)。exp038/exp046より単体では劣るが、Track G3のブレンド用アーキテクチャ多様性として評価予定 |
| 2026/07/21 08:01:29 | exp038_sigmafixed | `exp038_sigmafixed_submission.zip` | 0.6866381028699935 | holyholyholy | valid | exp038 strict + sigma_mode=fixed (predicted-sigma headなし)。exp046比 -0.00227、exp038単体比 -0.00253で**新green champion**。fold0/4ゲート両方改善済みで、5-fold実測でも一貫して効果あり |
| 2026/07/24 16:41:52 | exp047_sigmafixed | `exp047_sigmafixed_submission.zip` | 0.7001420620597125 | holyholyholy | valid | solar time特徴+sigma_fix。OOF 0.60747 (champion比-0.00081) だったが実測+0.01350の致命的悪化 — hemisphere×衛星one-hotによる訓練地域気候の暗記が疑われる。championから撤回 |
| 2026/07/24 16:42:13 | exp050_sigmafixed | `exp050_sigmafixed_submission.zip` | 0.6870176972903309 | holyholyholy | valid | split-window BTD+sigma_fix。OOF 0.60643 (champion比-0.00185、今セッション最良OOF) だったが実測+0.00038 — ノイズ帯内でほぼtransferせず |
| 2026/07/25 22:52:37 | exp056_seed456 | `exp056_seed456_submission.zip` | 0.6856908397786738 | holyholyholy | valid | exp056 seed456単体。OOF 0.61621 (最悪) だがLBはseed123より良い — OOF/LB順位不一致 |
| 2026/07/25 22:53:07 | exp056_seed123 | `exp056_seed123_submission.zip` | 0.6911129946327748 | holyholyholy | valid | exp056 seed123単体。OOF 0.60826 (2番目に良い) だがLB最悪 — seedノイズ~0.007の実証 |
| 2026/07/26 12:10:43 | exp056_seed_ensemble | `exp056_seed_ensemble_submission.zip` | 0.6840805442401546 | holyholyholy | valid | 3シード平均(42+123+456)。seed123に希釈されseed42とほぼタイ |
| 2026/07/26 12:11:07 | exp056_seed_ens_42_456 | `exp056_seed_ens_42_456_submission.zip` | 0.6827721114076882 | holyholyholy | valid | **2シード平均(42+456)= 新green champion**。両メンバーより良い、-0.00119 vs seed42単体 |
| 2026/07/27 10:14:15 | exp056_seed_ens_4best | `exp056_seed_ens_4best_submission.zip` | 0.6846399729749617 | holyholyholy | valid | 4シード平均(42+456+789+2024)。ens_42_456チャンピオン比 +0.00187で悪化 — スケールアップは効かず、round8計画のOOF所見(4/6-seed worse)をLBで確認 |
| 2026/07/27 10:14:30 | exp056_seed_ens_6 | `exp056_seed_ens_6_submission.zip` | 0.6848948764433148 | holyholyholy | valid | 6シード平均(42+123+456+789+1337+2024、全員)。ens_42_456チャンピオン比 +0.00212で悪化。seed123混入がens_ensemble同様に希釈 — seedアンサンブルのスケールアップは2シードでプラトー確定 |
| 2026/07/30 07:16:38 | exp064_effb3 | `exp064_effb3_submission.zip` | 0.6720097338314985 | holyholyholy | valid | 単体pretrained EfficientNet-B3、5-fold。fold0/4ゲートでは「mixed」判定(fold4良好/fold0悪化)だったが、実測5-fold LBは旧champion(exp056_seed_ens_42_456)比 **-0.01076** — ノイズ帯を大きく超える更新。**新green champion**。fold0/4ゲートがこの改善を予測できなかった点が重要 — pretrained backbone capacityの評価を見直す必要あり |
| 2026/07/30 07:17:24 | exp064_effv2s | `exp064_effv2s_submission.zip` | 0.6740019855472996 | holyholyholy | valid | 単体pretrained EfficientNetV2-S、5-fold。fold0/4ゲートでは旧championとほぼタイ判定だったが、実測5-fold LBは旧champion比 **-0.00877** |
| 2026/07/30 16:03:44 | exp065_champion_ensemble_causal | `exp065_champion_ensemble_causal.zip` | 0.6696622672831182 | holyholyholy | valid | exp056/effb3/effv2s/swin_lr2e4のアーキテクチャ多様4-wayアンサンブル(nested_blend.py outer-cross-fit重み: effb3=0.315/swin_lr2e4=0.315/exp056=0.27/effv2s=0.10)+causal平滑化。honest nested OOF -0.01191、overfitting_gap -0.00049、submission_gate.py GO判定。実測はexp064_effb3比 **-0.00235で新green champion** |
| 2026/07/30 16:04:15 | exp064_swin_lr2e4 | `exp064_swin_lr2e4_submission.zip` | 0.6704384663890527 | holyholyholy | valid | 単体pretrained Swin-Tiny(transformer)、lr=2e-4、5-fold。exp064系で唯一fold0/4ゲートを明確にPASSしていた候補で、実測でもexp064_effb3比 **-0.00157**で旧championを上回った。ゲート信頼性の知見を精緻化: 「明確なPASS」は依然シグナルとして機能する |
| 2026/07/22 01:02:31 | exp055 | `exp055_global_blend_causal.zip` | 0.6872098507591518 | holyholyholy | valid | g_eda/exp011のOOF最適48/52ブレンド(exp038_sigmafixed×exp040_metric) + g_eda/exp010のcausal平滑化/blur/衛星別threshold。OOF予測0.59984(-0.00868)に対し実測はexp038_sigmafixed単体比+0.00057の**悪化** — 深刻なOOF/LB逆転 |
| 2026/07/22 01:02:58 | exp055 | `exp055_global_blend.zip` | 0.6872601993829903 | holyholyholy | valid | 同ブレンド、平滑化なし版。OOF予測0.59982(-0.00870)に対し実測は単体比+0.00062の**悪化**。平滑化ありなし(causal版との差0.00005)はノイズ帯 — 逆転の原因はブレンド重みそのもの |
| 2026/07/24 17:58:34 | exp056 | `exp056_submission.zip` | 0.6839627937847801 | holyholyholy | valid | Mean-intensity×normalized-shape分解アーキテクチャ (g_eda/exp002のoracle-ladderが示した「残差の支配項はAMOUNT」への正面対応)。exp038_sigmafixed比 -0.00268で**新green champion**。事前にsubmission_gate.pyでGO判定 (OOF Δ-0.00576、80%CI `[-0.00920,-0.00259]`、18/20 locationで改善) — OOFの約47%がLBに転移し、健全な範囲 |
| 2026/07/30 23:08:29 | exp065_champion_ensemble_v2_causal | `exp065_champion_ensemble_v2_causal.zip` | 0.6684780268241325 | holyholyholy | valid | v1と同じ4-wayアンサンブル、後処理(causal smoothing tap重み・blur sigma・衛星別閾値)のみ再チューニング。v1比 **-0.00118で新green champion**。低自由度の変更で、OOF→LB転写も素直 |
| 2026/07/31 08:30:47 | exp064_effb4 | `exp064_effb4_submission.zip` | 0.6771412784573648 | holyholyholy | valid | 単体pretrained EfficientNet-B4(effb3からのスケールアップ)、5-fold。effb3比 **+0.00513悪化** — EfficientNet系統内の容量スケールアップはこの軸をEXHAUSTEDにする |
| 2026/07/31 08:31:17 | exp064_convnext_small_lr2e4 | `exp064_convnext_small_lr2e4_submission.zip` | 0.6793384249392268 | holyholyholy | valid | 単体pretrained ConvNeXt-Small、lr=2e-4、5-fold。exp064系全体で最下位。tiny版(fold0/4 tie)と合わせ、ConvNeXt系統はサイズ非依存でこのタスクとの相性が悪いと判断 |
| 2026/08/01 11:49:04 | exp064_pvt_v2_b0_lr2e4 | `exp064_pvt_v2_b0_lr2e4_submission.zip` | 0.672541575619076 | holyholyholy | valid | 単体pretrained PVTv2-B0(新backbone系統)、5-fold。5-fold OOFは今セッション単体最良(0.59617、effb3の0.59758を上回る)だったが、実測LBはexp064_effb3比 **+0.00053とノイズ帯内の実質タイ** — 0.001台のOOF差はLB順位を予測しないことが判明 |
| 2026/08/01 11:49:30 | exp064_effb3_seed456 | `exp064_effb3_seed456_submission.zip` | 0.6758692035492916 | holyholyholy | valid | effb3のseed456版。OOFはseed42比+0.00932悪化していたが、実測LB悪化はexp064_effb3比**+0.00386**(OOF差の約41%のみ転移)。exp056で確認済みのseedノイズがpretrained backbone系統でも再現。単体では使えないが、seed42+456の2シードアンサンブルが次候補 |
| 2026/08/01 11:53:18 | exp064_swin_small_lr2e4 | `exp064_swin_small_lr2e4_submission.zip` | (error) | holyholyholy | error | プラットフォーム側のinternal server errorで未採点(1回目)。再提出して確定 → 下記12:05:44の行を参照 |
| 2026/08/01 12:05:44 | exp064_swin_small_lr2e4 | `exp064_swin_small_lr2e4_submission.zip` | 0.6719466133572829 | holyholyholy | valid | 再提出で確定。単体pretrained Swin-Small(swin_lr2e4/Tinyからのスケールアップ)、5-fold。OOFはTiny比-0.00080改善だったが実測LBは**+0.00151悪化** — effb4/convnext_smallと同じ「OOF改善→LB悪化」逆転。**pretrained backbone容量スケールアップ軸はEfficientNet/ConvNeXt/Swin全系統でEXHAUSTED確定** |
| 2026/08/01 20:14:05 | exp065_champion_ensemble_6way_causal | `exp065_champion_ensemble_6way_causal.zip` | 0.6670330551889574 | holyholyholy | valid | exp056/effb3/effv2s/swin_lr2e4/pvt_v2_b0/swin_smallの6-wayアンサンブル。`nested_blend.py`のouter-cross-fit重み(pvt_v2_b0=0.2487最大/effb3=0.2035/swin_lr2e4=0.1938/swin_small=0.1615/exp056=0.1425/effv2s=0.05)+この6-way自身のOOFに再チューニングしたcausal smoothing。honest nested OOF 0.58418、v1比-0.00395。実測LBはv2比**-0.00145、v1比-0.00263で新green champion**。単体で勝てなかったpvt_v2_b0が最大ブレンド重みを取り、実際にLBを押し上げた — アーキテクチャ多様性仮説の2例目の実証 |

## Leaderboard Context (snapshot 2026-07-16, from the user)

| Rank | Team | Best Score |
| ---: | --- | ---: |
| 1 | Bull | 0.6347430711554105 |
| 2 | MahmoudElshahed | 0.6364063025228484 |
| 3 | syamoji141 | 0.6370789425592164 |
| 4 | Abdourahamane | 0.6382482536958467 |
| 5 | ouah7 | 0.6421310889779515 |
| 10 | motokimura | 0.6484923137595531 |
| 20 | alexandru | 0.6638379839406772 |
| **29** | **holys519 (us)** | **0.6706858062196032** |

Gap to #1 is 0.036; to top-10 is 0.022. The top pack has moved ~0.007 since 07-10.

## Leaderboard Context (snapshot 2026-07-10, from the user)

| Rank | Team | Best Score |
| ---: | --- | ---: |
| 1 | syamoji141 | 0.6421646651812066 |
| 2 | MahmoudElshahed | 0.6433509290178828 |
| 3 | hengck23 | 0.644324898564977 |
| 4 | BlueLock | 0.6457890107649674 |
| 5 | Bull | 0.6480379622758122 |
| 6 | ExaltedLAB | 0.6485607668756154 |
| 7 | motokimura | 0.6502917222272064 |
| 8 | ouah7 | 0.6506530845120508 |
| 9 | born | 0.6515104244663297 |

Key reference lines from the official discussion (`doc/discussion_insights.md`): the flat tile-mean
oracle ("the wall") scores 0.677 on train; perfect tile-mean + perfect wet/dry mask scores 0.594;
predict-all-zeros scores 0.746. exp033_w018_050 (0.6720) is below the wall. E-1
(`outputs/g_eda/exp002`) shows the dominant residual for exp016-018 is per-tile AMOUNT error,
not placement.

## Reference / Non-Primary Scores

| Experiment | Environment | Public RMSE | Source | Notes |
| --- | --- | ---: | --- | --- |
| exp001 | A100 test | 0.7937729717031525 | submission list 2026/07/07 11:20:13 | Reproduced on cloud, worse than local run |

## Observations

### exp056: an architectural bet that paid off, and confirms the toolkit isn't just a naysayer (2026-07-24)

`exp056` (mean-intensity × normalized-shape factorized architecture) is the first genuine
architecture change submitted in several rounds — prior candidates (exp047/exp050/exp055) were all
feature or blend micro-optimizations chasing sub-noise OOF deltas. It is motivated by
`g_eda/exp002`'s oracle-ladder finding that per-tile AMOUNT/intensity error, not spatial placement,
dominates this architecture family's residual, and factorizes the served prediction into a
separately-supervised scalar mean-intensity head times a normalized spatial shape head instead of
one joint per-pixel head. Result: **new green champion at 0.68396, -0.00268 vs exp038_sigmafixed**.

This is also the first candidate scored by `l_eda/exp005/submission_gate.py` *before* being folded
into this doc's narrative as a retrospective diagnosis: it returned **GO** (OOF Δ-0.00576, 80% CI
`[-0.00920, -0.00259]` excluding zero, gain diffuse across 18/20 locations, no concentration flag),
and about 47% of the OOF gain transferred to the real LB delta — a healthy, unremarkable transfer
ratio, unlike exp047/exp050's near-zero or inverted transfer. Together with the exp055 negative
result below, this is useful evidence about what the toolkit is and isn't good for: it correctly
passed a real, generalizable architecture improvement instead of flagging everything as suspect, and
it correctly rejected two illusory feature-engineering gains — but it also shows a false GO on a
blend whose problem was train-vs-eval distribution shift, not sampling noise. The practical lesson
matches the original diagnosis this round started from: further investment in genuine architecture
changes (not feature/blend micro-tuning) is where the remaining ~10 days are best spent.

- **exp038 (0.68916) is the strict/green champion**: exp011 strict比でOOF −0.01967、
  Public −0.03407。exp018/exp035との0.003–0.007級の順位差は不安定なため、
  `outputs/g_eda/exp007/TRANSFER_AUDIT.md` の群別・bootstrap監査を判断基準にする。
- **exp039_4src_joint_patched (0.66191) is the overall tracked best**, but it is red because
  it uses overlap patching; its raw 0.67896 counterpart is amber. exp033_w018_050 was the
  earlier blend milestone: mixing exp018 at 50% into
  `equal_016_017` before the patch gains −0.00266 over exp026 — exp018 adds real diversity.
  The OOF-optimal mixture (weights, per-satellite variants, blur/threshold) is being computed
  in `g_eda/exp003`; `g_experiments/exp036` serves its recommendation.
- exp026 (0.67465): exp024 `equal_016_017` (0.69193) + overlap patch. Patch value on the
  blend = −0.01728 (vs −0.01847 on exp009 in exp014).
- **exp018 (0.69295) is the best single model**, in line with its best OOF (0.6093). The
  OOF→LB deltas are consistent: exp018−exp017 = −0.0068 LB vs −0.0070 OOF.
- **exp016 vs exp017 inverts on LB** (0.69776 vs 0.69974) relative to OOF (0.6186 vs 0.6163).
  Δ≈0.002 both ways — inside the E-3 noise band (residual std 0.0033). Treat the two as tied.
- **exp027's seed-family blends hurt** (0.6807 / 0.6849, both worse than exp026's 0.6746):
  exp025 seed checkpoints include weak folds (e.g. seed123 fold3 tile 1.02) and equal-type
  weighting dilutes the blend. exp039 harvest must weight by per-checkpoint OOF and drop weak
  members, not blend everything equally.
- **Blending exp009 in hurts** (0.6961/0.6940 vs 0.6919 without) — consistent with exp009's
  worse OOF (0.6239). Blend membership should track OOF quality.
- Next blend candidates: exp033/exp034 ladders mixing exp018 into `equal_016_017` (zips built);
  E-3 pairs now include exp016/017/018 solo scores for a stronger CV→LB regression.

### exp055 green blend: OOF/LB inversion (2026-07-22)

`g_eda/exp011` computed an OOF-optimal 48/52 blend of `exp038_sigmafixed` (OOF 0.60852) ×
`exp040_metric` (OOF 0.60772), giving blend OOF **0.59982** — a −0.00870 gain over
`exp038_sigmafixed` solo, far outside this project's established noise band (~0.004–0.005).
Both the raw blend and the causal-smoothed variant (`g_eda/exp010`'s coefficients, tuned for
`exp038_sigmafixed` solo, not re-fit for the blend) were submitted:

| Submission | OOF | Public RMSE | vs exp038_sigmafixed solo (0.68664) |
| --- | ---: | ---: | ---: |
| exp055_global_blend_causal | 0.59984 | 0.68721 | **+0.00057 (worse)** |
| exp055_global_blend (no smoothing) | 0.59982 | 0.68726 | **+0.00062 (worse)** |

**Neither beats the solo champion — the −0.0087 OOF gain did not transfer at all; it inverted.**
Using this project's own historical OOF→LB regression (`LB ≈ 1.268×OOF − 0.080`, fit on 13 red-era
pairs), blend OOF 0.59982 would predict LB ≈ 0.6808 — the actual 0.6873 is ~0.006 worse than even
that regression's already-conservative estimate. This is a materially larger inversion than the
single-model amber case seen earlier (`exp038_features`, OOF −0.0070 vs LB +0.0029).

**Working hypotheses** (neither yet confirmed — flagged for follow-up, not closed):

1. **Blend-weight overfitting to in-sample OOF.** `g_eda/exp011`'s first implementation fits the
   48/52 weight directly against the same OOF tiles used to report the −0.0087 gain, with no outer
   cross-fit/nested holdout (the two-source case only has "trivial/full ladder" search tiers — see
   `optimize_blend.py`). `doc/research_survey_v3_2026-07-16.md` Phase 1 already flagged that blend
   weight learning should use an outer cross-fit; this was simplified away for the first exp055
   pass. A weight fit this way can look arbitrarily good on the fold-holdout OOF while encoding
   structure specific to the *train* locations' fold split, which does not carry over to the
   (non-overlapping) eval locations — the same structural gap E-4/fold-anatomy already documents
   as this competition's dominant source of OOF↛LB slippage.
2. **exp040_metric's error structure may not generalize the same way exp038_sigmafixed's does.**
   Its distinct loss (tile-RMSE-shaped, `metric_weight=0.6`) could correlate with wetness/regime in
   a way that looks complementary on train folds (each model's mistakes partly cancel the other's on
   the *same* fold distribution) without being complementary on eval's different regime mix —
   consistent with exp040_metric's own solo LB (0.69552) sitting further behind exp038's solo LB
   (0.68916) than their OOF gap alone would suggest.

**Decision**: `exp038_sigmafixed` (0.68664) remains the green champion and the submission of record.
`exp055`'s blend is not adopted. **Before re-running the harvest with more sources** (exp047/050/
051/052/053/054, as their fold gates clear), `g_eda/exp011`'s weight search must be redone with an
outer cross-fit (fit weights on a subset of folds/locations, score on the held-out remainder,
repeated) rather than a single in-sample fit on the full OOF — this is the same discipline
`g_eda/exp003` used successfully for the (now-red) exp016/017/018 blends and should not have been
skipped here. `g_eda/exp010`'s causal-smoothing coefficients should also be re-tuned against
whatever blend eventually replaces this one, not carried over from the exp038_sigmafixed-solo fit.

## Template For Future Submissions

Add new submissions to both `Current Best` and `Submission Log`.

| Submitted at | Experiment | Submission | Public RMSE | User/Team | Status | Memo |
| --- | --- | --- | ---: | --- | --- | --- |
| YYYY/MM/DD HH:MM:SS | expXXX | `expXXX_submission.zip` | 0.0000000000000000 | holyholyholy | valid | short description |
