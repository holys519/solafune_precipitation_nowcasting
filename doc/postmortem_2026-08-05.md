# 敗因分析（多角的ポストモータム） 2026-08-05

*目的: 最終提出確定(`exp065_champion_ensemble_6way_causal`, Public 0.6670330552 / Private
0.6631893564)を受けて、実験データ・discussion・自己監査ドキュメントを横断し、上位との
ギャップがどこから生まれたかを角度別に整理する。新しい実験は行わず、既存記録の再解釈のみ。*

## 0. 結論の要約

単一の敗因はない。少なくとも5つの独立した要因が重なっている:
(1) タスク自体が20↔18地点で重複ゼロの分布シフト問題であり、これは実行力で埋めきれない
構造的ギャップを作る、(2) 自分たちのOOF検証は20地点しかなく統計的な解像度が粗いため、
本来もっと早く分かるはずだった知見(pretrained encoder)を3週間近く見送った、(3) 序盤〜中盤
に投じた時間の相当割合が「洗い出してみたらLBで消える/反転する」損失設計・特徴量エンジニア
リングに費やされた、(4) アンサンブルの多様性軸をアーキテクチャに求めたが、独立した強豪
(Bull)は「情報の多様性」軸で勝っており、こちらの軸は自分たちで一度試して棄却していた、
(5) 一定期間、他チームの規約違反(successor row/overlap patch)がPublic LBを歪めていた可能性
があり、その間のギャップは自分たちの手法の劣位を正しく反映していない。

## 1. 構造的要因: そもそも勝ちにくいタスク設計だった

- 訓練20地点・評価18地点は**重複ゼロ**(`discussion/discussion_all.md` §2.3)。LWIR平均値が
  GOESで-15.5、Meteosatで+13.4ずれるなど、train/evalは物理的に別分布。地理を暗記するモデル
  は構造的に負ける設計。
- `doc/discussion_insights.md` §1「0.677の壁」: タイル平均を完璧に当てても0.677止まり。
  トップ16はタイル**内**のどこに雨があるかを当てている。序盤の自分たちの戦略(exp009系の
  isotonic calibration等)はこの壁の**内側**の量的キャリブレーションに投資しており、そもそも
  勝敗を決める軸(空間配置)に序盤は乗っていなかった。
- `doc/twin_fold_and_trap_audit_2026-07-26.md`「Frontier calibration」: 独立参加者
  (MahmoudElshahed)の見立てでは検証可能な誠実な解は0.63–0.65。自分たちの最終防衛ライン
  (0.68277当時)は「その推定が正しければ」0.03–0.05の未発見ギャップがあるが、「情報限界説
  (0.68が実質天井)」も同程度にありうる、と**未解決のまま終了**。これは実行の失敗ではなく
  知識の限界であり、正直に「分からない」として記録しておく。

## 2. 検証(OOF)設計の限界 — 自分たちの目が粗かった

- 5-foldは20地点しか裏付けがなく、fold0はわずか2地点(`doc/task_tickets.md` Round 7)。
  0.005未満のOOF差はfold構成のノイズと見分けがつかない。
- `doc/twin_fold_and_trap_audit_2026-07-26.md`: `atlantic_coast↔florida`(37%重複)・
  `bihar↔dhaka`(13%重複)が同一GPMグリッドのため**ピクセル同一ラベル**を持つツインペアで、
  championのseed42分割は両方ともfoldをまたいでいた → 絶対OOFは0.60252ではなく実質
  0.61–0.62程度に楽観バイアスがかかっていた(相対比較には影響小、絶対値の信頼性が低い)。
- `doc/task_tickets.md` I-002: `nested_blend.py`のouter-cross-fitでも exp055 blendは
  GO判定だったのに実測LBは悪化。20地点内のリサンプリングでは**train↔eval分布シフト自体を
  検出できない**という、ツールの原理的な限界が実測で確認されている。
- `doc/plan/round8_campaign_2026-07-27.md`「2026-07-30 update」: fold0/4の2-fold簡易ゲート
  は`exp064_effb3`/`exp064_effv2s`(最終的に単体でchampion更新した2本)を「mixed/tie」と誤判定
  し、**あわや破棄するところだった**。たまたま「作ったから一応5-foldまで回した」おかげで
  発見された。ゲートが信頼できるのは「明確なpass」の時だけで、「mixed/tie=価値なし」ではない
  と後から学んだ — この教訓が出たのが07-30、締切間際。

## 3. モデリング面: 損失設計の「うさぎ穴」に時間を使いすぎた

- `doc/discussion_insights.md` §2の数理的知見(hurdle分解、log-normal serving、wet画素のみ
  学習)に基づき、`two_head_rain`・quantile head・tail re-weighting(`pos_weight`/
  `bce_pos_weight`)など一連の損失設計をRound 6で試している。
- しかし`doc/oof_lb_transfer_by_category_2026-07-25.md`が明確に示す通り、**アーキテクチャ変更
  はOOF→LBが忠実に転写するが、特徴量追加・損失設計・ブレンド重みのフィットは反転/相殺する**
  (exp040_metric: OOF改善なのにLB+0.00888悪化。exp047_sigmafixed: プロジェクト最悪の
  +0.01350悪化)。
- 独立に、rank1のBullも同じ結論に達している(`discussion/discussion_all.md`「Bull: Tips for
  Improving Your Score」): *"loss engineering is an easy rabbit hole with little return, spend
  the time on EDA/ensembling instead"*。plain MSE (log1p space)がhurdle/Tweedie/Charbonnier/
  重み付けに勝ったと明言。
- **問題は「損失設計が無駄だった」ことではなく、その結論に自分たちで辿り着くまでに
  Round 6〜7(exp047/exp050/exp055/exp040等、複数ラウンド)を要したこと**。Bullは早期にこれを
  切り上げてEDA/アンサンブルに全振りしている。唯一のジャンル横断的な勝ち筋(アーキテクチャ
  変更=exp056)が実際に投入されたのは07-24で、締切(実質08-03頃)まで残り10日というタイミング
  だった(`doc/public_scores.md`の該当observationが自ら「further investment in genuine
  architecture changes... is where the remaining ~10 days are best spent」と明記)。

## 4. 見過ごされていた最大のレバー: pretrained encoderの後回し

- 実は**プロジェクト最初の実験**(`doc/exp001_retrospective.md`、7月上旬)の時点で「pretrained
  encoder(efficientnet-b0)を使うつもりだったがcustom CompactUNetで代替した」ことと、その
  ABテストの必要性が明記されている(item 7、優先度は最後に格下げ)。
- 実際にこの軸(`exp064`)に着手したのは07-27〜07-28、結果が出て championを更新したのが
  07-30。**最終的に一番効いた変更の仮説は競技開始直後から手元にあったが、実装着手まで約3週間
  かかっている**。理由は当時「非RGB54chの入力にpretrained重みを素直に使えない」という技術的
  ハードルと、他の軸(損失設計・特徴量)を先に消化する優先順位付けだった。
- 皮肉なことに、07-27時点の`doc/pretrained_backbone_findings_2026-07-27.md`の初回ゲート結果
  は「capacity is NOT the bottleneck」という誤った結論だった(2-foldゲートの解像度不足、
  §2参照)。もし更に早く着手していれば、この誤判定を修正して本命に育てる時間的余裕も
  大きかったはずだが、実際には締切直前に判明したため、11-way以降の追加探索(effb1/effb2/
  regnety016/densenet121)は全滅という結果に終わっている(`doc/public_scores.md` 追記17-19)。

## 5. アンサンブル戦略の分岐 — 多様性の軸を読み違えた可能性

- 自分たちの最終アンサンブル(exp065, 6-way)は**同じ入力・異なるアーキテクチャ(feature
  extractor)**による多様性(exp056 from-scratch CNN + effb3/effv2s/swin/pvt/swin-small)。
  `doc/plan/round8_campaign_2026-07-27.md`はこれを「architecture-diverse」と呼び、実際に
  championを更新した唯一の有効打だった。
- 一方でBull(rank1)は明確にこう述べている: *"The diversity that works is 'different
  information' (longer past-time context, alternative temporal sampling), not 'different
  architectures' looking at the same inputs (those saturate and get zero ensemble weight)."*
  つまりBullの経験では**自分たちが採った軸(アーキテクチャ多様性)はゼロ重みになりがち**で、
  情報量(時間文脈など)を変える方が効くと明言している。
- 自分たちは逆に、情報量側の軸(`exp063`: predecessor 2h・207ch長期履歴)を試して**両fold
  悪化で棄却**している(`doc/submission_registry.md`)。つまり両チームは対照的な軸で
  それぞれ成功/失敗しており、この矛盾は未解決。可能性としては (a) 自分たちの長期履歴の実装
  が悪かった、(b) データセットの時間分解能(30分window)ではそもそも長期履歴に乗る情報が
  Bullのタスク設定ほど無い、(c) 両方とも正しく、単に「まだ試していない情報多様性の実装」が
  他にある、のいずれか切り分けられていない。**ここは実験不足で結論が出ていない、明確な
  伸びしろ候補**。

## 6. 学習レシピの差: 5-fold CVアンサンブル vs 全データ再学習

- 自分たちの最終submission群は一貫して「5-foldそれぞれで学習したモデルの予測をfoldごとの
  OOFで検証し、そのままアンサンブル/ブレンドに使う」設計(`g_experiments/exp065/README.md`)。
- Bullは明確に別のレシピを勝ち筋として報告している: *"Full-data retraining (all sites, no
  fold split) consistently improved LB after the config was fixed via CV."* つまりCVは
  ハイパーパラメータ確定のためだけに使い、**最終提出は20地点全部を使った単一モデル**で行う、
  という設計。
- 20地点しかない小規模データで、各fold学習が実質16地点分の情報しか使わない自分たちの方式は、
  データ効率の観点で損をしている可能性がある。**この一手は最後まで検証されなかった**
  (grep上、`full.data`/`no.fold`retrainに該当する実験記録は見当たらない)。

## 7. 特徴量トラップ: geography由来の記憶

- `exp047_sigmafixed`(緯度・半球・local solar time等の位置特徴)はOOFではむしろ改善(-0.00081)
  だったのに、LBは**プロジェクト最悪の+0.01350悪化**(`doc/oof_lb_transfer_by_category
  _2026-07-25.md`)。原因は「hemisphere × satellite one-hot」が学習地域の気候を暗記する
  ショートカットになったこと。
- これは第2回答者への回答(本スレッド)でも運営が最終確定した「地名→座標のクローズドフォーム
  特徴は許可、気候区分・標高等は禁止」という線引きと整合する事後診断であり、**自分たちの
  プロジェクトはこの罠を自力で先に発見し、以後同種の特徴を「geography由来は無罪推定しない」
  という運用ルールに格上げしていた**(同docの運用ルール2)。運営の最終回答が出た8月時点では
  この教訓はすでに織り込み済みで、今回のchampionにはgeography特徴は使われていない。

## 8. フェアネス環境: ルール違反勢との比較不能性

- `doc/competition_rules.md`: 2026-07-20〜07-22にかけて運営が「successor row入力」
  「overlap patch」「evaluation画像を跨ぐreverse engineering」を明確にredと確定させたが、
  **検証は入賞/メダル圏のみ・コンペ終了後のみ**で、それ以外のPublic LBスコアは一切訂正
  されない設計だった。
- 自分たち自身も一時期successor row(exp009系, +0.01相当)やoverlap patch(exp014)を試して
  おり、これらは`doc/submission_registry.md`で明確にredとして最終提出から除外している。
  つまり**期間中盤まで見えていたPublic LBの一部(自分たちを含む)は正当な手法の実力を反映
  していなかった**。07-16スナップショットの「1位との差0.036」は、この意味で汚染された
  比較である可能性が高い。
- 結果として、「なぜ07-16時点で29位だったか」を額面通りの実力差として受け取るのは正確では
  ない。真に比較可能なのは、運営裁定後にredを除外して green のみで再構築した後の順位だが、
  そのタイミングでの公開順位スナップショットはローカルに残っていない(データギャップ)。

## 9. 良かった点（対比のため）

- 検証ツール(`submission_gate.py`, `nested_blend.py`, bootstrap CI, twin-fold audit)は、
  試みられた3件の疑わしい昇格のうち2件(`exp047_sigmafixed`, `exp050_sigmafixed`)を実際に
  NO-GOとして事前に弾いている。
- 最終championのPublic→Private乖離は-0.0038で、Bullが見積もった±0.008のノイズ帯に収まり、
  shakeupなし。これは検証プロセスが「実質的に誠実な」提出を選べていたことの傍証。
- コンプライアンス面は一貫してクリーン(`scripts/verify_causal_replay.py`で40/40 bit-identical
  確認済み)。ルールが最終確定した後の混乱・失格リスクを負っていない。

## 10. もしもう一度やるなら（優先度順）

1. **pretrained encoderのAB比較をRound 1〜2で即着手する**(exp001時点で仮説はあった)。
   非標準チャネル数へのstem適応は前例のある小さな技術課題であり、後回しにする理由が薄い。
2. **損失設計・特徴量エンジニアリングの「まず1周試して見切る」判断を早める**。
   `doc/oof_lb_transfer_by_category`のルール(アーキテクチャ変更は信頼、特徴/損失は要LB確認)
   を、後追いでなく最初のラウンドから運用する。
3. **fold0/4簡易ゲートを"tie/mixed"の時に切り捨てない**運用に最初から変える。特に
   pretrained encoder系は解像度不足になりやすいと分かった以上、次回は最初から5-foldへ
   進める閾値を緩める。
4. **アンサンブルの多様性軸を「アーキテクチャ」と「情報(時間文脈・サンプリング)」の両方で
   並行して探索する**。exp063の棄却が実装起因か本質起因か、次回は切り分けてから判断する。
5. **CV確定後に全データ再学習した単一モデルを、5-fold CVアンサンブルと必ず比較する**。
   今回は未検証のまま終わった。
6. Twin-site(ピクセル重複)を最初からgroup keyに畳み込んだ分割にする — 今回は発見が
   07-26と遅く、絶対OOFの信頼性を最後まで割り引く必要があった。
