# exp066: Temporal architecture for causal history (ConvLSTM / attention fusion)

Implements the round8 plan's queued "Next generations" item: replace exp063's crude channel-stack
of predecessor rows with a **shared per-frame encoder + explicit temporal fusion**, instead of
concatenating every frame's raw channels into one flat blob and hoping the first conv layer learns
the temporal structure from scratch.

## Motivation

exp063 tested the "legal long causal history" axis (predecessor rows at T-30/T-60/..., context_rows
2/3/4) by naive channel-stacking, and it failed at every depth tried:

| exp | context_rows | fold0 | fold4 | verdict |
| --- | ---: | ---: | ---: | --- |
| exp063 (cr1 baseline) | 1 | 0.28159 | 0.58503 | baseline |
| exp063_cr2 | 2 | 0.28302 | 0.59143 | worse (fold4 +0.0064, outside noise) |
| exp063_cr3 | 3 | 0.28838 | 0.59237 | worse |
| exp063 (cr4) | 4 | 0.28474 | 0.59410 | worse |

This was closed as "history doesn't help, information axis exhausted" -- but the round8 plan's own
07-30 finding (fold0/4 gate underestimating the pretrained-backbone axis) is a reminder that a
negative result can be an artifact of *how* the information was fed to the model, not proof the
information itself is useless. Crude concatenation gives the network T*17 raw channels with no
inductive bias about which channels belong to the same time step or what order they're in --
exactly the kind of structure a from-scratch conv stack has to rediscover per training run, on only
20 non-overlapping train locations.

## Architecture (`model.py`, `TemporalFusionFactorizedUNet`)

- Each of the `T = context_rows * max_observations` frames (16 raw satellite channels + 1
  observed/padded mask channel + the 3-channel satellite one-hot, broadcast identically to every
  frame) is run independently through a **shared** 4-level encoder (`enc1..enc4`, identical
  `ConvBlock` design to `FactorizedMeanShapeUNet` -- same capacity per level, so any tile_rmse delta
  vs exp063_cr2 is attributable to the fusion mechanism, not extra encoder capacity).
- Temporal fusion happens **only at the bottleneck** (cheapest place to run a sequence model): the
  per-frame bottleneck-input feature maps are reordered into strict causal (oldest -> newest) order
  (`_causal_order` -- a row-group flip, since dataset.py's raw layout is newest-row-first) and fused
  via one of:
  - `model.temporal_fusion: convlstm` (`config_convlstm_cr2.yaml`) -- a single ConvLSTM cell
    stepped across the T frames, final hidden state = fused representation. Explicitly causal/
    recurrent.
  - `model.temporal_fusion: attention` (`config_attention_cr2.yaml`) -- a learned per-frame scalar
    score (1x1 conv + global pool + linear), softmax over T, weighted sum. Permutation-invariant,
    no recurrence.
- Skip connections (`e1..e4`) come from the **current** (newest) frame only, exactly like the
  single-frame champion's own skip connections -- fine spatial detail from the most recent
  observation, abstract "trend from history" from the fused bottleneck. This keeps the decoder
  (`dec4..dec1`) and every head (`rain_head`/`shape_head`/`mean_intensity_mlp`/`aux_mask_head`)
  byte-identical to `FactorizedMeanShapeUNet`, copied verbatim.
- `dataset.py`, `losses.py`, `train.py`, `inference.py`, `smoke_test.py` are all copied **unmodified**
  from exp063: the model reconstructs the per-frame view internally from the known flat-channel
  layout (`expected_in_channels`), so the input wire format, loss, and training loop don't change at
  all. Only `model.py` (new architecture + `build_model` dispatch entry
  `temporal_fusion_factorized`) is new.

## Configs

Both use `context_rows: 2` (current row + T-30, in_channels: 105) -- the **same information budget
as `exp063_cr2`**, so this is a clean architecture-only comparison against that specific crude-stack
failure, not a different (and confounded) history depth.

- `config_convlstm_cr2.yaml`
- `config_attention_cr2.yaml`

`train.batch_size: 32` (down from exp063's 128): the shared per-frame encoder runs `B * 6` forward
passes per step instead of `B * 1`, so this keeps peak activation memory comparable to the existing
5-fold jobs. `early_stopping_patience: 20` set from the start (exp064_effb4/exp064_convnext_small_lr2e4
postmortem -- checkpointing is already synchronous on every validation improvement, so this only
saves wasted GPU time).

## Reading the result

- If **both** arms beat `exp063_cr2` (0.28302/0.59143) but still lose to the cr1 baseline
  (0.28159/0.58503): architecture helped but this information axis is still not worth it at cr2 --
  informative, but not a submission candidate.
- If either arm **beats the cr1 baseline outright**: this reopens the history axis as a live track,
  worth extending to cr3/cr4 with the winning fusion mechanism.
- If both arms still lose to `exp063_cr2`: the crude-stack failure was NOT primarily an
  architecture problem -- the history axis itself is genuinely uninformative for this task, a much
  stronger and more defensible closure of the axis than exp063's original result alone.
- ConvLSTM vs attention: the round8 plan's own gate-reliability lesson (07-30) says a "mixed"/"tie"
  fold0/4 verdict is not trustworthy evidence of "no value" for architecture changes -- run to full
  5-fold LB regardless of the fold0/4 read, per that same discipline, before writing either arm off.
