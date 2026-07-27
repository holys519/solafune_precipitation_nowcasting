# Pretrained-backbone (exp064) findings — 2026-07-27

Tested the long-deferred "from-scratch is underpowered" hypothesis (ticket L-002, never started):
exp056's factorized head on 5 timm ImageNet-pretrained encoders (in_chans=54 stem auto-adapted),
gated fold0/4 vs the from-scratch champion exp056 (fold0 0.28159 / fold4 0.58503).

| Encoder | params | fold0 | fold4 | verdict |
| --- | ---: | ---: | ---: | --- |
| exp056 (from-scratch, champion) | ~5M | 0.28159 | 0.58503 | baseline |
| tf_efficientnetv2_s | 20M | 0.28208 (+0.0005) | 0.58531 (+0.0003) | **~tie both folds (noise)** |
| efficientnet_b3 | 10M | 0.28833 (+0.0067) | 0.58135 (-0.0037) | mixed (trades folds) |
| resnet34 | 21M | 0.28699 (+0.0054) | 0.59281 (+0.0078) | both worse |
| convnext_tiny | 28M | (NaN) | (NaN) | diverged (AMP+lr, epoch ~?) |
| swin_tiny | 28M | (NaN@ep25) | (NaN) | diverged |

## Conclusion: capacity is NOT the bottleneck

The best pretrained backbone (efficientnetv2_s) only TIES the from-scratch 5M champion; none beats
it. Pretrained encoders fit the train set better (effv2s train tile_rmse ~0.36 vs exp056's higher)
but do not generalize better to held-out folds. This confirms the **generalization-limited /
train-eval-distribution-shift** view: adding ImageNet-pretrained capacity improves train fit, not
out-of-region generalization. It answers "why weren't we using a giant encoder?" empirically — here
it doesn't help. Per the OOF-transfer rule (architecture changes transfer faithfully), a gate tie
means an LB tie, so none of these is worth a solo submission.

Transformer/modern backbones (convnext, swin) diverged to NaN under AMP + lr=0.001 + LayerNorm;
a lower-LR/grad-clip/warmup rescue is possible but NOT pursued, since even the clean-training
pretrained CNNs don't beat the champion.

## Residual value: architecture diversity for ensembling

Both Bull and MahmoudElshahed report that *genuinely diverse* ensembling (different information/
architecture, not seeds) is what moves out-of-region error. Our seed ensemble plateaued (same
architecture, correlated errors: 2-seed 42+456 = 0.68277 best, more seeds worse). effv2s/effb3 are
tie-quality but structurally very different feature extractors (ImageNet efficientnet vs from-scratch
CNN), so exp056 x effv2s(x effb3) is the untested *architecture-diverse* blend that could decorrelate
errors where seeds could not. effv2s + effb3 advanced to full 5-fold for this purpose (not as solo
champions). Final blend to be built + LB-checked; champion floor remains exp056_seed_ens_42_456
(0.68277).
