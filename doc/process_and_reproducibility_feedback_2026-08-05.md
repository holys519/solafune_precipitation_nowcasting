# 実験プロセスへのフィードバックと再現性ログの改善案 2026-08-05

*`doc/postmortem_2026-08-05.md`(なぜ負けたか)の続編。ここでは「実験の回し方そのもの」と
「再現性・実装ログを後から確認しやすくする仕組み」に焦点を当てる。新しい実験は行わず、
既存のディレクトリ構成・ドキュメント・登録漏れの実例から評価する。*

## Part A: 実験プロセスへのフィードバック

### 良かった点

1. **段階的ゲート(fold0/4 → 5-fold)+ 数値的なGO/NO-GO基準**を自作し、勘に頼らない昇格判断
   の仕組みを持っていた(`l_eda/exp005/submission_gate.py`, `g_eda/exp011/nested_blend.py`)。
   実際に3件の危うい昇格のうち2件(`exp047_sigmafixed`, `exp050_sigmafixed`)を事前に弾いている。
2. **負の結果を隠さず記録する文化**。`exp059`/`exp060`/`exp062`/`exp063`など、棄却された実験も
   `doc/submission_registry.md`に理由付きで残っており、同じ失敗を繰り返さないための資産になって
   いる。
3. **知見を運用ルールへ即座に格上げするループ**があった。`doc/oof_lb_transfer_by_category
   _2026-07-25.md`や`doc/twin_fold_and_trap_audit_2026-07-26.md`のように、発見した教訓を
   「今後こう判断する」という具体的な運用ルールに変換して次のラウンドに適用していた。

### 改善余地

1. **安くて確度の高い実験の先送り**。`doc/exp001_retrospective.md`(7月上旬)の時点で
   「pretrained encoderを本来使うつもりだった」と明記されていたのに、実際の着手(`exp064`)は
   07-27〜07-28。約3週間、最も期待値の高い仮説が優先順位表の下位に置かれ続けた。判断自体
   (非RGB54chスタムへの適応が技術的に未知数だった)は妥当だが、「不確実なら小さく早く検証する」
   のではなく「確実な軸を先に消化してから」という順番になっていた。
2. **簡易ゲート(fold0/4)自体の較正が遅れた**。`doc/plan/round8_campaign_2026-07-27.md`が示す
   通り、pretrained backbone系の候補をこのゲートは「mixed/tie」と誤判定し、最終的にchampionを
   更新した`exp064_effb3`/`effv2s`を危うく切り捨てるところだった。ゲートの信頼性(どの軸で
   fold0/4が本物のシグナルになり、どの軸でならないか)は既知の「勝ち」(`exp056`)を通して
   早期に検証しておくべきだった。
3. **損失設計のうさぎ穴からの脱出が遅かった**。「アーキテクチャ変更だけがOOF→LBを裏切らない」
   という結論(`doc/oof_lb_transfer_by_category`)に達するまでにRound 6〜7、複数の実験
   (`exp040`/`exp047`/`exp050`/`exp055`)を要した。外部のdiscussion(Bull: *"loss engineering
   is an easy rabbit hole with little return"*)は同じ結論をもっと早く言っており、外部発信を
   早い段階で参照する運用があれば、この期間のコストを削減できた可能性がある。
4. **実験の"登録漏れ"が実際に発生した**。`exp052`(未来フレーム補助head)と`exp053`(自己回帰)は
   5-fold OOF/ゲートまで完走していたのに、`doc/submission_registry.md`への反映が数週間放置され、
   **たまたま2026-08-01の監査で「発見」された**(`doc/submission_registry.md`該当行に明記)。
   実験が終わったのに、終わったこと自体がチームの記録に載っていない — これは今回最も具体的で
   再発防止しやすい失敗パターン(対策はPart B)。
5. **並行設計が投資回収されなかった**。`doc/research_pipeline_design.md`(07-24)は
   `src/precip_nowcast`という「クリーンな再現性重視パッケージ」を設計し、seed/依存バージョン
   記録・fold/location manifest同梱などの立派な"publication checklist"まで定義しているが、
   実際にLBを動かした`exp056`/`exp064`/`exp065`はすべて従来の`g_experiments/expNNN`アドホック
   構成のまま進み、**`src/precip_nowcast`はどの実験からも参照されずgit上に取り残された**
   (`git log -- src/`は初期コミットと`exp064`コミットの2件のみ)。理想的な再現性設計を書く
   コストを、実際に使われているパイプラインの改善に向けた方がよかった可能性がある。
6. **ディレクトリ衛生の乱れ**。`g_experiments/exp064/`配下に無関係な`slurm-exp056-*.out/err`
   ログが56個混入している。ジョブ名とディレクトリ名がずれた事故で、後から監査する人が
   「このディレクトリで実際に何が走ったか」を誤読しかねない。

## Part B: 再現性確認・実装ログの改善案

### 現状の構造的ギャップ

- `doc/experiment_tracking_design.md`(07-09)は「1実験1行、gitで差分追跡できるCSVレジストリ
  (`doc/experiment_registry.csv`)」を設計したが、**このファイルは一度も作成されていない**
  (未実装のまま競技終了)。
- 代わりに機能していたのは`scripts/plot_experiment_scores.py`が生成する`doc/score_history/`
  (CSV群+SVG+`REPORT.md`)。ただし**最終生成は2026-07-30時点で止まっており**、その後の
  重要な提出(`exp065_champion_ensemble_6way_causal`08-01、`exp064`系の残り08-02〜08-04)は
  この自動集計に一切反映されていない。「自動化はしたが、運用として最後まで回し続けられ
  なかった」。
- `doc/submission_registry.md`は手動保守のMarkdown表で、上述の通り実際に登録漏れが発生した。
  自然言語の備考欄にステータス(green/amber/red、ゲート結果、5-fold完走かどうか)が混在して
  おり、機械的な横断検索(grep以外)がしづらい。
- `doc/competition_rules.md`が要求する再現性マニフェスト(pretrained checkpointのURL・
  version・license・取得日・SHA-256・load箇所)は、実際には`g_experiments/exp064/model.py`の
  コード内コメントに散文で書かれているだけで、**構造化されたmanifestファイルが存在しない**。
- どのsubmissionがどのgit commitで作られたかを記録した箇所がゼロ(`submission_registry.md`,
  各expの`README.md`とも`git_commit`列/記載なし)。後からcommit hashを確定する手段が
  タイムスタンプの突き合わせしかない。

### 改善提案(優先度・実装コストが低い順)

1. **「実験終了」をパイプラインが自動検出する**。5-foldの最終foldジョブ完走を
   `singularity_run.sh`側で検知し、その最後のステップとして`doc/submission_registry.md`
   (またはB-2のCSV)に1行を自動追記する。exp052/53のような「終わったのに誰も知らない」を
   構造的に防ぐ、07-09の設計書がまさに提案していた内容を今度こそ実装する。
2. **手動Markdown表 → 機械可読なCSV/JSONレジストリへ一本化**。`doc/experiment_tracking_
   design.md`のスキーマ(run_id, git_commit, kind, architecture, oof指標, public_rmse,
   status)をそのまま`doc/experiment_registry.csv`として実装し、`doc/submission_registry.md`
   はこのCSVから自動生成される人間向けビューに位置づけを変える(手で二重管理しない)。
3. **submission生成時にJSON manifestを機械的に書き出す**。`make_submission.py`の最後に、
   config.yamlの内容・checkpointパスとハッシュ・現在のgit commit hash・(pretrained重みが
   あれば)取得元URL/SHA-256/ライセンス/取得日を1つのJSONへdumpする関数を追加するだけで、
   `doc/competition_rules.md`が要求するReproducibility項目をコードから機械的に満たせる。
   現状のREADME散文コメントによる記録より、監査時の検索・突合が圧倒的に速くなる。
4. **`doc/score_history`の再生成を「思い出したら手で実行」から「提出のたびに必須」へ格上げ**
   する。`REPORT.md`冒頭に生成日時が入っている設計自体は良いので、そこに「最新のPublic
   submissionより古い場合は警告」を足すだけで、今回のような08-01〜08-04の空白を機械的に
   検知できるようになる。
5. **並行設計を始める前に移行コストを見積もる/使わないなら明記する**。`src/precip_nowcast`
   のような「あるべき姿」の設計は、実際のパイプラインをそこへ移行する計画とセットでない限り
   `README.md`に「参考設計・未使用」と明記し、監査する人が「これが実際に動いたコードだ」と
   誤認しないようにする。
6. **ディレクトリ/ジョブ名の一致をチェックする軽量lint**を`singularity_run.sh`投入前に
   噛ませる(sbatchジョブ名が実行ディレクトリ名の接頭辞と一致しているかだけの正規表現チェック
   で十分)。今回のexp064配下のexp056ログ混入のような事故を機械的に検出できる。

### まとめ

どの提案も新規の重い仕組みではなく、**すでに設計だけは存在していたが実装/運用まで至らな
かったもの**(`experiment_registry.csv`、score_historyの継続運用、manifest記録)を実際に
完走させることが中心になる。次回は「設計ドキュメントを書いた時点で満足しない」ことと、
「実験終了の記録をパイプラインの一部にして人手に依存させない」ことの2点が、再現性監査を
楽にする上で最も投資対効果が高い。
