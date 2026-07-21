# Public Scores

Last updated: 2026-07-20

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
rank 1つずつに `[ELIGIBLE]` / `[RED]` を付記した。**2026-07-20の裁定後、初のgreen専用結果が出た:
exp046_causal_smoothed (0.68891) がexp038 (0.68916) をわずかに更新し、新しいeligible green
champion**。exp040_metric単体 (0.69552) はexp038単体より劣るが、Track G3ブレンド用の
アーキテクチャ多様性の2本目としての価値は別途評価する。

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
| 12 | exp027 | `exp027_equal_all_patched.zip` | 0.6849224439171961 | 2026/07/13 12:02:26 | valid | **[RED]** successor-row sources + patch |
| 13 | exp035 | `(recorded in E-3 audit)` | 0.6860146267326392 | — | valid | **[RED — 2026-07-20確定]** context_rows: 2 |
| 14 | exp046 | `exp046_causal_smoothed_submission.zip` | 0.6889118106607066 | 2026/07/20 01:31:56 | valid | **[ELIGIBLE — new green champion]** exp038 + causal-only temporal smoothing (center=0.85/prev=0.15, next=0, untuned). Beats exp038 solo by -0.00025 -- confirms causal smoothing (permitted 2026-07-20) has real, if small, value even with un-tuned weights |
| 15 | exp038 | `exp038_submission.zip` | 0.6891638997287517 | 2026/07/18 06:01:30 | valid | **[ELIGIBLE]** strict current-row-only green model, context_rows: 1; superseded by exp046 |
| 16 | exp024 | `exp024_equal_016_017.zip` | 0.6919274860606568 | 2026/07/12 05:13:36 | valid | **[RED — 2026-07-20確定]** exp016/017 blend, both context_rows: 2 |
| 17 | exp040_metric | `exp040_metric_submission.zip` | 0.6955180267195701 | 2026/07/20 01:32:36 | valid | **[ELIGIBLE]** standalone green model, architecturally distinct from exp038 (metric_weight=0.6 tile-RMSE-shaped loss). Weaker solo than exp038/exp046, but intended as the 2nd model for Track G3 green-blend diversity, not a solo champion |

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

## Template For Future Submissions

Add new submissions to both `Current Best` and `Submission Log`.

| Submitted at | Experiment | Submission | Public RMSE | User/Team | Status | Memo |
| --- | --- | --- | ---: | --- | --- | --- |
| YYYY/MM/DD HH:MM:SS | expXXX | `expXXX_submission.zip` | 0.0000000000000000 | holyholyholy | valid | short description |
