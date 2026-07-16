# Public Scores

Last updated: 2026-07-16

This file tracks public/valid leaderboard scores for the Solafune precipitation nowcasting
competition. Metric is RMSE, so lower is better.

Sources:

- `doc/exp001_retrospective.md`
- `doc/research_survey.md`
- Solafune submission list copied by the user on 2026-07-08 / 07-09 / 07-10 / **07-16 (full list)**

## Current Best

| Rank | Experiment | Submission | Public RMSE | Submitted at | Status | Notes |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | `exp036_per_satellite_sm0p25_blur1_thr0p2_patched.zip` | 0.6661746681900441 | 2026/07/16 07:33:53 | valid | 上に時間平滑化 (0.25/0.30/0.45) を追加 — **current best**。OOF予測−0.0038に対し実測−0.0045 |
| 2 | `exp036_per_satellite_blur1_thr0p2_patched.zip` | 0.6706858062196032 | 2026/07/16 06:48:24 | valid | OOF衛星別重み (goes 15%/him 55%/met 60% exp018) + blur σ1.0 + thr 0.2 + patch |
| 3 | `exp033_w018_050_patched.zip` | 0.671989922822016 | 2026/07/16 10:41:11 | valid | 50% exp024 `equal_016_017` + 50% exp018, then overlap patch |
| 4 | `exp026_submission.zip` | 0.6746506841387548 | 2026/07/13 12:01:55 | valid | exp024 `equal_016_017` blend + exp014 overlap patch |
| 5 | `exp027_half016_half017family_patched.zip` | 0.6806568162687938 | 2026/07/13 12:02:46 | valid | 50% exp016 + 12.5%×4 exp017 seed family + patch — **worse than exp026**: seed checkpoints dilute quality |
| 6 | `exp027_equal_all_patched.zip` | 0.6849224439171961 | 2026/07/13 12:02:26 | valid | Equal 5-way blend + patch — worse still |
| 7 | `exp024_equal_016_017.zip` | 0.6919274860606568 | 2026/07/12 05:13:36 | valid | Unpatched exp016/017 50/50 blend (best unpatched) |
| 8 | `exp018_submission.zip` | 0.6929495140301676 | 2026/07/16 12:57:23 | valid | High-res localization — **best single model**, consistent with best OOF 0.6093 |
| 9 | `exp024_blend_20_40_40.zip` | 0.693975964307325 | 2026/07/12 05:10:40 | valid | 20/40/40 exp009/016/017 |
| 10 | `exp024_equal_009_016_017.zip` | 0.6961199095679 | 2026/07/12 05:11:35 | valid | Adding exp009 to the blend hurts |
| 11 | `exp014_submission.zip` | 0.6968727727408199 | 2026/07/10 12:44:13 | valid | Tile-Overlap GPM Copy Patch on exp009 (G-022) |
| 12 | `exp016_submission.zip` | 0.6977629323809645 | 2026/07/11 11:23:41 | valid | Hurdle log-normal head |
| 13 | `exp017_submission.zip` | 0.6997414980565597 | 2026/07/11 11:24:23 | valid | Physics channels — LB order vs exp016 inverts their OOF order (Δ0.002 < noise) |
| 14 | `exp015_submission.zip` | 0.7096658388930687 | 2026/07/10 08:47:38 | valid | Isotonic OOF calibration on exp009 (G-027a) |
| 15 | `exp009_submission.zip` | 0.7153438899106017 | 2026/07/09 12:38:39 | valid | Successor-Row Frames |
| 16 | `exp011_submission.zip` | 0.7232307883574975 | 2026/07/09 12:44:58 | valid | Satellite Adapter Two-Head |
| 17 | `exp008_submission.zip` | 0.7250185237499447 | 2026/07/09 12:36:08 | valid | Official Metric + Drizzle Post-Processing |
| 18 | `exp004_submission.zip` | 0.7252533726905589 | 2026/07/08 02:25:26 | valid | Two-Head Rain Detection + Amount Regression |
| 19 | `exp010_submission.zip` | 0.7348731115909746 | 2026/07/09 12:40:03 | valid | Data Cleanup Two-Head |
| 20 | `exp007_submission.zip` | 0.7362157342148196 | 2026/07/09 01:20:50 | valid | Multi-exp equal-weight ensemble |
| 21 | `exp005_submission.zip` | 0.7445524878914139 | 2026/07/08 02:36:58 | valid | Temporal fusion |
| 22 | `exp006_submission.zip` | 0.7450324204392412 | 2026/07/08 02:48:44 | valid | Satellite adapter |
| 23 | `exp002_submission.zip` | 0.7479569114058262 | 2026/07/08 07:14:08 | valid | A100_exp002 |
| 24 | `exp003_submission.zip` | 0.7522576632294679 | 2026/07/08 09:40:50 | valid | A100_exp003 |
| 25 | `exp001_submission.zip` | 0.7531995875751526 | 2026/07/07 04:29:43 | valid | Local baseline |

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

- **exp033_w018_050 (0.67199) is the current best**: mixing exp018 at 50% into
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
