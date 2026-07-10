# Public Scores

Last updated: 2026-07-10

This file tracks public/valid leaderboard scores for the Solafune precipitation nowcasting
competition. Metric is RMSE, so lower is better.

Sources:

- `doc/exp001_retrospective.md`
- `doc/research_survey.md`
- Solafune submission list copied by the user on 2026-07-08
- Solafune submission list copied by the user on 2026-07-09

## Current Best

| Rank | Experiment | Submission | Public RMSE | Submitted at | Status | Notes |
| ---: | --- | --- | ---: | --- | --- | --- |
| 1 | exp014 | `exp014_submission.zip` | 0.6968727727408199 | 2026/07/10 12:44:13 | valid | Tile-Overlap GPM Copy Patch (post-processing on exp009 base, G-022) |
| 2 | exp015 | `exp015_submission.zip` | 0.7096658388930687 | 2026/07/10 08:47:38 | valid | Isotonic OOF Calibration on exp009 checkpoints (G-027a) |
| 3 | exp009 | `exp009_submission.zip` | 0.7153438899106017 | 2026/07/09 12:38:39 | valid | Successor-Row Frames |
| 4 | exp011 | `exp011_submission.zip` | 0.7232307883574975 | 2026/07/09 12:44:58 | valid | Satellite Adapter Two-Head |
| 5 | exp008 | `exp008_submission.zip` | 0.7250185237499447 | 2026/07/09 12:36:08 | valid | Official Metric + Drizzle Post-Processing |
| 6 | exp004 | `exp004_submission.zip` | 0.7252533726905589 | 2026/07/08 02:25:26 | valid | Two-Head Rain Detection + Amount Regression |
| 7 | exp010 | `exp010_submission.zip` | 0.7348731115909746 | 2026/07/09 12:40:03 | valid | Data Cleanup Two-Head |
| 8 | exp005 | `exp005_submission.zip` | 0.7445524878914139 | 2026/07/08 02:36:58 | valid | Temporal fusion |
| 9 | exp006 | `exp006_submission.zip` | 0.7450324204392412 | 2026/07/08 02:48:44 | valid | Satellite adapter |
| 10 | exp002 | `exp002_submission.zip` | 0.7479569114058262 | 2026/07/08 07:14:08 | valid | A100_exp002 |
| 11 | exp003 | `exp003_submission.zip` | 0.7522576632294679 | 2026/07/08 09:40:50 | valid | A100_exp003 |
| 12 | exp001 | `exp001_submission.zip` | 0.7531995875751526 | unknown | valid | Local baseline from `doc/exp001_retrospective.md` |

## Submission Log

| Submitted at | Experiment | Submission | Public RMSE | User/Team | Status | Memo |
| --- | --- | --- | ---: | --- | --- | --- |
| unknown | exp001 | `exp001_submission.zip` | 0.7531995875751526 | unknown | valid | Compact U-Net, simple concat, plain MSE |
| 2026/07/08 02:25:26 | exp004 | `exp004_submission.zip` | 0.7252533726905589 | holyholyholy | valid | Two-Head Rain Detection + Amount Regression |
| 2026/07/08 02:36:58 | exp005 | `exp005_submission.zip` | 0.7445524878914139 | holyholyholy | valid | Temporal fusion |
| 2026/07/08 02:48:44 | exp006 | `exp006_submission.zip` | 0.7450324204392412 | holyholyholy | valid | Satellite-specific adapter |
| 2026/07/08 07:14:08 | exp002 | `exp002_submission.zip` | 0.7479569114058262 | holyholyholy | valid | A100_exp002 |
| 2026/07/08 09:40:50 | exp003 | `exp003_submission.zip` | 0.7522576632294679 | holyholyholy | valid | A100_exp003 |
| 2026/07/09 12:36:08 | exp008 | `exp008_submission.zip` | 0.7250185237499447 | holyholyholy | valid | Official Metric + Drizzle Post-Processing |
| 2026/07/09 12:38:39 | exp009 | `exp009_submission.zip` | 0.7153438899106017 | holyholyholy | valid | Successor-Row Frames |
| 2026/07/09 12:40:03 | exp010 | `exp010_submission.zip` | 0.7348731115909746 | holyholyholy | valid | Data Cleanup Two-Head |
| 2026/07/09 12:44:58 | exp011 | `exp011_submission.zip` | 0.7232307883574975 | holyholyholy | valid | Satellite Adapter Two-Head |
| 2026/07/10 08:47:38 | exp015 | `exp015_submission.zip` | 0.7096658388930687 | holyholyholy | valid | Isotonic OOF Calibration on exp009 checkpoints (G-027a) |
| 2026/07/10 12:44:13 | exp014 | `exp014_submission.zip` | 0.6968727727408199 | holyholyholy | valid | Tile-Overlap GPM Copy Patch (post-processing on exp009 base, G-022) |

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
predict-all-zeros scores 0.746. Our exp014 (0.6969) sits essentially at the wall — the gap to the
top is within-tile localization, not tile-level amount accuracy. Round 4 (exp016-exp020,
`doc/task_tickets.md` G-030..G-034) targets this.

## Reference / Non-Primary Scores

| Experiment | Environment | Public RMSE | Source | Notes |
| --- | --- | ---: | --- | --- |
| exp001 | A100 test | 0.7937729717031525 | `doc/research_survey.md` | Reproduced on cloud, worse than local run |

## Observations

- exp014 is the current best public result, confirming the tile-overlap GPM-copy patch
  (`doc/tile_overlap_discovery.md`, G-022): applying it as pure post-processing on top of the exp009
  submission improves public RMSE by 0.0184711171697818 (0.7153438899106017 -> 0.6968727727408199),
  the single largest jump recorded in this table. This validates the FFT cross-correlation overlap
  detection and the train-train (atlantic_coast/florida) bit-exact copy check used to design it.
- exp015 (isotonic OOF calibration, G-027a, `g_experiments/exp015`) improves over the exp009 base by
  0.0056780510175330 (0.7153438899106017 -> 0.7096658388930687). This is a genuine, real
  improvement despite an OOF-side approximate check (`oof_sample_metrics.csv` tile-level pred_max)
  suggesting the isotonic curve barely corrects the heavy-rain tile-peak underestimation (mean
  pred_max moved 2.13 -> 2.10 against a target mean of 7.90) -- the win most likely comes from
  reining in over-confident mid/high false-alarm pixels rather than fixing peak amplitude, which
  the tile-peak proxy check wasn't measuring. exp015 was not yet stacked with exp014's tile-overlap
  patch; doing so (pointing `exp014/apply_overlap.py` at exp015's raw predictions instead of
  exp009's) is the next likely best-single-submission candidate.
- exp009 was the previous best public result. Successor-row frames are the strongest *modeling*
  signal so far, improving over the previous exp004 anchor by 0.0099094827799572 RMSE.
- exp011 improves over exp004, suggesting the satellite-adapter plus two-head combination is useful
  even though exp006 alone did not beat exp004.
- exp008 slightly improves over exp004, so the official-metric/drizzle post-processing path remains
  worth keeping as a low-risk post-processing layer.
- exp010 improves over exp005/exp006/exp002/exp003 but is worse than exp004/exp008/exp011/exp009,
  so its cleanup assumptions need OOF diagnostics before becoming the default.
- exp002 improved over exp001 only modestly on public RMSE, so future changes should be judged with
  OOF diagnostics as well as public score.
- exp007 public score is not recorded here yet.

## Template For Future Submissions

Add new submissions to both `Current Best` and `Submission Log`.

| Submitted at | Experiment | Submission | Public RMSE | User/Team | Status | Memo |
| --- | --- | --- | ---: | --- | --- | --- |
| YYYY/MM/DD HH:MM:SS | expXXX | `expXXX_submission.zip` | 0.0000000000000000 | holyholyholy | valid | short description |
