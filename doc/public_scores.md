# Public Scores

Last updated: 2026-07-09

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
| 1 | exp009 | `exp009_submission.zip` | 0.7153438899106017 | 2026/07/09 12:38:39 | valid | Successor-Row Frames |
| 2 | exp011 | `exp011_submission.zip` | 0.7232307883574975 | 2026/07/09 12:44:58 | valid | Satellite Adapter Two-Head |
| 3 | exp008 | `exp008_submission.zip` | 0.7250185237499447 | 2026/07/09 12:36:08 | valid | Official Metric + Drizzle Post-Processing |
| 4 | exp004 | `exp004_submission.zip` | 0.7252533726905589 | 2026/07/08 02:25:26 | valid | Two-Head Rain Detection + Amount Regression |
| 5 | exp010 | `exp010_submission.zip` | 0.7348731115909746 | 2026/07/09 12:40:03 | valid | Data Cleanup Two-Head |
| 6 | exp005 | `exp005_submission.zip` | 0.7445524878914139 | 2026/07/08 02:36:58 | valid | Temporal fusion |
| 7 | exp006 | `exp006_submission.zip` | 0.7450324204392412 | 2026/07/08 02:48:44 | valid | Satellite adapter |
| 8 | exp002 | `exp002_submission.zip` | 0.7479569114058262 | 2026/07/08 07:14:08 | valid | A100_exp002 |
| 9 | exp003 | `exp003_submission.zip` | 0.7522576632294679 | 2026/07/08 09:40:50 | valid | A100_exp003 |
| 10 | exp001 | `exp001_submission.zip` | 0.7531995875751526 | unknown | valid | Local baseline from `doc/exp001_retrospective.md` |

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

## Reference / Non-Primary Scores

| Experiment | Environment | Public RMSE | Source | Notes |
| --- | --- | ---: | --- | --- |
| exp001 | A100 test | 0.7937729717031525 | `doc/research_survey.md` | Reproduced on cloud, worse than local run |

## Observations

- exp009 is the current best public result. Successor-row frames are the strongest signal so far,
  improving over the previous exp004 anchor by 0.0099094827799572 RMSE.
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
