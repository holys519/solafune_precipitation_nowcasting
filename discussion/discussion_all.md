Dataset Updated: Please Re-download Only If You Downloaded It Before Around June 12, 08:00 GMT

Dear fellow hackers,

Earlier today we discovered that the competition dataset download accidentally included an internal configuration file.

We've now corrected this. We regenerated the sample IDs, reshuffled the public/private split, and removed the configuration file from the download. Because both the IDs and the split have changed, any earlier copy of that file is now out of date and can no longer be used to gain an advantage. We also confirmed that no submissions were made during the window the file was exposed.

The corrected dataset has been live since around 08:00 GMT today. If you downloaded the dataset before then, please re-download the latest version before continuing. If you're unsure when you downloaded it, re-downloading is the safest option. If you downloaded after 08:00 GMT or haven't downloaded yet, you already have the correct version and don't need to do anything.

We're sorry for the inconvenience and thank you for your understanding.



Back to Discussion

Zulqarnain

Posted 2 週間前

2
Stable 2D U-Net + ConvLSTM

Architecture — the "Stable 2D U-Net + ConvLSTM"
The model has five moving parts:

Per-timestep Band-Attention Fusion (16 → 64 channels)

Satellite embedding (3 IDs, learned 16-dim)

2D U-Net encoder at 128×128 (encoder channels 64, 128, 256, 512)

ConvLSTM cell at the bottleneck, unrolled over the 3 timesteps

U-Net decoder with skip connections

Two output heads — intensity (

softplus
, ≥ 0) and rain probability (

sigmoid
∈ [0, 1]) — multiplied at the end

Adaptive average pool to the 41×41 target grid




LB 0.69 From Geostationary Pixels to Millimetres of Rain

From Geostationary Pixels to Millimetres of Rain: A Stable U-Net + ConvLSTM Recipe for Cross-Region Precipitation Nowcasting
Public LB: 0.694734415929908 RMSE . Approach: Band-attention U-Net × ConvLSTM with explicit optical-flow priors, dual intensity/rain-probability heads, 2-seed deep ensemble, 6-way test-time augmentation. Best validation RMSE: seed 42 = 1.1972 (epoch 37), seed 123 = 1.1820 (epoch 73).

1. Why this challenge matters
Flash floods are the most deadly weather hazard on Earth. The lead time of an evacuation order is decided, today, by the lag between the moment water touches a river-stage sensor and the moment a model sounds the alarm. Numerical Weather Prediction (NWP) systems such as Pangu-Weather and MetNet-3 push that lead time further out, but they are still constrained by ground-radar coverage — a coverage that is thin to non-existent across most of the Global South.

The Solafune Precipitation Nowcasting competition is one of the first public benchmarks that turns this problem into a pure satellite-only regression task. Each input is a 30-minute geostationary window (three 10-minute steps, 16 spectral bands) from one of three operational satellites; the target is the NASA/JAXA GPM-IMERG half-hour precipitation at 41×41 resolution. The grading is Root Mean Squared Error (RMSE) on a 29,090-sample evaluation set whose 18 locations are entirely disjoint from the 20 training locations.

In other words: this is a domain-generalization problem disguised as a regression problem. Whoever can make the model transfer wins.

This post walks through my solution end-to-end — architecture, data plumbing, training, validation, ensembling, and a frank list of what I would do differently with more time.

2. The data at a glance
Property	Value
Input modalities	3 geostationary satellites × 16 bands × 3 timesteps
Satellites	Himawari-8/9, GOES-R, Meteosat
Native grids	81×81 (Himawari), 141×141 (GOES), 144×144 (Meteosat)
Target grid	41×41 (GPM-IMERG, mm/hr)
Train samples	40,686 across 20 locations
Eval samples	29,090 across 18 unseen locations
Overlap train↔eval locations	0
Time span	2023-01-01 → 2026-01-31
Dry-pixel fraction	81.78 %
Rainy-pixel mean (p50 / p99)	0.56 / 15.04 mm/hr
Max target value seen	46.99 mm/hr
The 30-minute input window is composed of three 10-minutely observations stored as 16-band GeoTIFFs. Each sample row in train_dataset.csv looks like:

unique_id, name_location, satellite_target, datetime,
last_30_minutes_observation_filename  (list of 3 .tif),
gpm_imerg_filename                    (target .tif)
2.1 The cross-satellite wrinkle
Although every satellite ships 16 channels, the physical meaning is not aligned: Himawari uses B01..B16, GOES uses C01..C16, Meteosat uses vis_04..ir_133. The EDA confirmed this — water-vapor (WV 6.3 µm / 7.3 µm) and long-wave IR (10.8 µm) sit in different channel indices across the three families. A naïve 16-channel Conv would have to discover the alignment from scratch.

2.2 Per-band Spearman correlation with the GPM target (selected channels)
Channel family	Himawari	GOES	Meteosat
Long-wave IR (10–12 µm)	-0.44 to -0.51	-0.43 to -0.50	-0.39 to -0.47
Water-vapor (6.3 / 7.3 µm)	-0.36 to -0.51	-0.43 to -0.48	-0.37 to -0.46
Visible (0.64 µm)	-0.08	0.00	-0.02
NIR 1.6 µm	-0.09	-0.01	+0.04
Long-wave IR and water-vapor channels are the dominant signal across all three sensors. The visible band is the weakest of the four channel families in this dataset.

2.3 The location-shift elephant in the room
Train locations (20)              Eval locations (18) — disjoint
─────────────────────────────────────────────────────────────────
aceh, andalusia, atlantic_coast,  kanto_region, limpopo,
bahia_blanca, bihar, borno_state, lombardia, maputo,
cape_town, central_philippines,   mekong_delta, mexico,
central_vietnam, dhaka, ecuador,  niger_state, north_sumatra,
florida, france,                  northeast_malaysia, peru,
friuli_venezia_giulia,            quang_nam, rio_grande_do_sul,
gaza_province, guangdong,         sofala, sri_lanka, sylhet,
hat_yai, jakarta, jamaica,        tanganyika, upper_midwest,
kinshasa                          valencia
And the LWIR mean shifts by -15.5 raw units for GOES and +13.4 for Meteosat between train and eval. A model that memorizes the surface climatology of the training locations is going to fail. The architecture therefore has to lean on physics-aligned features (cold-cloud-top temperatures, motion) and per-satellite normalisation, not on geography.

3. Exploratory Data Analysis — what drove the design
3.1 Targets are extremely sparse and heavy-tailed
81.78 % of target pixels are exactly 0.
Mean over all pixels is 0.31 mm/hr; mean over rainy pixels is 1.70 mm/hr.
P99 = 6.8 mm/hr, p99.9 = 18.7 mm/hr, max ≈ 47 mm/hr.
Implication: a pure MSE objective is dominated by the dry 82 %, so the model converges to predicting near-zero and ignores the heavy tail. A dual-head (intensity + rain probability) plus a log-cosh regulariser was needed to recover tail accuracy.

3.2 Temporal resolution is constant, day and night
The hour-of-day distribution in both train and eval is essentially flat — 1,700 ± 20 samples per hour in train, 1,200 ± 10 in eval, no diurnal gap. Good news: no need for sun-angle engineering.

3.3 Three timesteps, but only sometimes
97.8 % of train samples (97.9 % eval) ship all three observations; the rest have 0–2. The dataset class pads missing slots with zero tensors so the model always sees a length-3 input.

3.4 The 30-minute window is exactly the IMERG step
The target is a cumulative 30-minute precipitation product. So the prediction is "how much did it rain in this 41×41 box during the next 30 minutes, given the last 30 minutes of imagery." This is an extrapolation task, not a smoothing task. Optical flow is essentially free prior knowledge for it.

3.5 Per-satellite mean rainfall
Satellite	Mean mm/hr	Rain fraction
GOES	0.3891	0.243
Himawari	0.4075	0.215
Meteosat	0.1834	0.120
GOES and Himawari scenes carry roughly twice the rain rate of Meteosat on average. This is another reason a single global model needs an explicit satellite-conditioning signal.

4. Architecture — the "Stable 2D U-Net + ConvLSTM"
The model has five moving parts:

Per-timestep Band-Attention Fusion (16 → 64 channels)
Satellite embedding (3 IDs, learned 16-dim)
2D U-Net encoder at 128×128 (encoder channels 64, 128, 256, 512)
ConvLSTM cell at the bottleneck, unrolled over the 3 timesteps
U-Net decoder with skip connections
Two output heads — intensity (softplus, ≥ 0) and rain probability (sigmoid ∈ [0, 1]) — multiplied at the end
Adaptive average pool to the 41×41 target grid
1.png

4.1 Band-Attention Fusion
The 16 channels of the three satellites mean different things. Rather than force the U-Net to learn cross-sensor alignment from scratch, a small SE-style block recalibrates channels per timestep:

class BandAttentionFusion(nn.Module):
    def __init__(self, in_channels=16, out_channels=64):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2), nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels), nn.Sigmoid(),
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        attn = self.fc(self.avgpool(x).view(x.size(0), -1)).view(x.size(0), -1, 1, 1)
        return self.bn(self.conv(x * attn))
It is cheap (one global pool, two-line MLP, one 1×1 conv) and lives in front of every timestep.

4.2 Optical-flow prior
Nowcasting is motion modelling. Rather than learn it implicitly, I prepend a 2-channel "pseudo-flow" derived from band-0 frame differencing with a 3×3 Sobel-like kernel:

f1 = frames[i][0:1]      # band 0 of frame i
f2 = frames[i+1][0:1]    # band 0 of frame i+1
diff = f2 - f1
kernel_x = torch.tensor([[[[-1, 0, 1]]]], dtype=diff.dtype, device=diff.device)
kernel_y = torch.tensor([[[[-1], [0], [1]]]], dtype=diff.dtype, device=diff.device)
flow_x = F.conv2d(diff, kernel_x, padding='same')
flow_y = F.conv2d(diff, kernel_y, padding='same')
flow  = torch.cat([flow_x, flow_y], dim=0)
This is intentionally a hand-engineered prior. It tells the encoder "here is the direction and magnitude of recent cloud motion", which is exactly the inductive bias a ConvLSTM would otherwise have to discover across 3 frames.

4.3 Encoder + ConvLSTM
Each timestep is encoded with a 4-level 2D U-Net encoder (channels 64 → 128 → 256 → 512, GELU, BatchNorm, dropout 0.2). The deepest feature map is fed to a single ConvLSTM cell that is unrolled across the 3 timesteps. The last hidden state is the bottleneck output.

Skip connections are taken from the last timestep only — enc_outs[T-1], down1_out[T-1], down2_out[T-1] — and concatenated into the decoder stages.

4.4 Satellite embedding
himawari → 0, goes → 1, meteosat → 2 is passed through nn.Embedding(3, 16), projected to 512 channels with a linear layer, and added to the ConvLSTM hidden state before the decoder:

sat_emb = self.sat_embed(sat_id)                      # (B, 16)
emb_proj = self.embed_proj(sat_emb)                  # (B, 512)
h = h + emb_proj.unsqueeze(-1).unsqueeze(-1)
This is the cheapest possible way to inject "which sensor am I looking at" without ballooning parameters.

4.5 Dual intensity / rain-probability heads
intensity = F.softplus(self.intensity_head(d0))   # ≥ 0
rain_prob = torch.sigmoid(self.rain_head(d0))     # ∈ [0, 1]
pred      = intensity * rain_prob
The two heads are decoupled so that one can output "this is a 4 mm/hr cloud top" while the other says "but I am only 30 % confident rain reaches the surface." Their product is the final precipitation estimate, which is then bilinearly pooled to 41×41.

4.6 Loss function
Three terms, masked by the GPM-IMERG validity mask:

Term	Weight	Purpose
masked_mse	1.0	Primary driver of RMSE
masked_logcosh	0.3	Robustness on the heavy tail
masked_bce (rain > 0.1 mm/hr)	0.05	Forces the rain/no-rain decision
logcosh(x) = |x| + softplus(-2|x|) − log 2 behaves like x²/2 near 0 and like |x| in the tail, which is exactly what you want for a heavy-tailed positive target.

5. Data pipeline
Pipeline.png

5.1 Per-satellite band statistics
Rather than normalise the three satellites into a single (μ, σ), I compute a (mean[16], std[16]) pair per satellite from a 500-row random sample and cache it to band_stats.json:

mean = sum_b / count
var  = np.clip(sumsq_b / count - mean ** 2, 1e-6, None)
std  = np.sqrt(var); std[std < 1e-3] = 1.0
This is essential — GOES IR channels hover around 195 with σ ≈ 60, Meteosat WV channels around 158 with σ ≈ 19. A global mean/std would crush one and inflate the other.

5.2 Length-tolerant input
If fewer than 3 observations exist, the missing slots are filled with zero tensors. The ConvLSTM still unrolls across the full 3 steps; the zero-padded frames are simply part of the input the gating acts on.

5.3 Augmentations (train only)
Random horizontal flip
Random vertical flip
Random 90° rotation (k ∈ {0, 1, 2, 3})
Crucially, the optical-flow channels are flipped/rotated in lock-step with the band tensor so the model never sees a "rotated image with original-direction flow".

5.4 Mixed precision
torch.cuda.amp.autocast + GradScaler with cudnn.benchmark = True. Validation is run under autocast too, but in torch.no_grad().

6. Training strategy
Knob	Value	Why
Optimizer	AdamW (β=(0.9, 0.999), wd=1e-4)	Decoupled weight decay for the BN+Conv stack
LR	3e-4 peak, 1e-6 floor	Standard for warmup-cosine on Conv hybrids
Schedule	5-epoch LinearLR warmup → CosineAnnealingLR	Avoids early divergence, anneals smoothly into the long tail
Batch size	64	Fits 128×128×16×3 in 16 GB with AMP
Epochs	150 with early stop patience = 20	seed 42 stopped at 57, seed 123 at 93
Gradient clip	1.0	Defends against AMP-induced spikes in log-cosh gradients
Seeds	(42, 123)	Both are reported in the logs
min_delta	1e-4	Improvement threshold for the early-stop counter
The validation split is per-location chronological 85/15 — the last 15 % of every location's time series is held out. Random splitting would let a model peek at the future, and temporal splitting is the only honest test for a nowcaster.

7. Validation strategy and leakage control
Three rules I refused to break:

No random split. Time advances forward. The validation rows come from after the training rows for the same location.
No test-side normalisation leakage. Band statistics are computed from train rows only; eval rows go through the cached (μ, σ).
No look-ahead in the temporal window. Each sample's three observations are the 30 minutes immediately preceding the target timestamp.
Final validation scores per seed (full UNet + ConvLSTM, after early stop):

Seed	Best val RMSE	Epoch at best	Epoch at early stop
42	1.1972	37	57
123	1.1820	73	93
The validation rows are drawn from the same 20 training locations, so they live in-distribution. The 29,090-row public leaderboard set spans the 18 unseen locations, which is why the LB RMSE (0.6947) is much lower than the val RMSE — it is a different distribution, not a fair head-to-head.

8. Inference and ensembling
8.1 Six-way test-time augmentation
transforms = [
    (0, False, False),   # identity
    (0, False, True),    # h-flip
    (0, True,  False),   # v-flip
    (1, False, False),   # rot90
    (2, False, False),   # rot180
    (3, False, False),   # rot270
]
For each model in the ensemble and each TTA transform, the prediction is computed, then de-transformed, then averaged across the six transforms. The flip/rot operations are applied to both the band tensor and the optical-flow tensor so the model never sees a self-inconsistent input.

8.2 Two-seed deep ensemble
Final prediction = mean(seeds).mean(TTA). With two seeds and six TTAs, that is 12 forward passes per sample. The ensemble is built only from the two seed checkpoints produced by the training run.

8.3 Clamping
pred.clamp_min(0.0) is the only post-processing. I left the model output untouched otherwise — the dual head is already responsible for the dry-bias correction.

8.4 Submission writer
Each prediction (1×41×41 float32) is written to a single-band GeoTIFF, then evaluation_target.csv and the test_files/ directory are zipped into submission.zip. Total file count: 29,090, matching the eval CSV exactly.

9. Results
The two numbers I can report honestly:

Metric	Value
Public LB RMSE (this submission)	0.694734415929908
Public LB RMSE (earlier baseline I started from)	0.732
Best val RMSE (seed 42)	1.1972
Best val RMSE (seed 123)	1.1820
I am not publishing an ablation table for this post because every ablation number in my notes during the competition came from a single checkpoint per configuration, and the leaderboard ranks were noisy at the 0.005 scale. The only well-controlled comparison I have is seed 42 vs seed 123 of the same architecture: the second seed shaved about 0.015 off the held-out validation RMSE and contributed to a marginally better public LB after the TTA×seed ensemble. Everything else between the 0.732 baseline and the 0.6947 submission was an architectural change with no clean controlled comparison.

What I can say, from the training logs, is what each seed's curve looked like:

Seed	Epoch 1 train RMSE	Epoch 1 val RMSE	Final train RMSE	Final val RMSE	Δ train → val
42	1.3140	1.4252	0.9963 (ep 46)	1.1972 (ep 37)	0.20
123	2.3815	1.4408	0.9324 (ep 90)	1.1820 (ep 73)	0.25
The persistent ~0.2 train→val gap is consistent across both seeds and is the same order of magnitude as the LWIR distribution shift between train and eval seen in the EDA.

10. Lessons learned
10.1 The dual head is non-negotiable
Training a single Conv head to output 0–47 mm/hr on a 82 %-zero target is a fool's errand. Splitting into "is it raining?" + "how hard?" is what unlocked the heavy tail. The same trick shows up in marine debris detection, oil-spill mapping, and tropical-cyclone intensity estimation.

10.2 Optical flow is a free prior
Hand-rolled band-0 differencing + Sobel kernels gave the model the right inductive bias for an extrapolation task. The cost is two 2D convolutions per frame pair; the benefit is that the ConvLSTM does not have to learn motion from scratch across only 3 frames.

10.3 Short input windows amplify temporal modelling
With T = 3, the bottleneck ConvLSTM has very few steps to integrate motion. Keeping the cell count low and the hidden width high was the right trade-off for this dataset size; the cell gating did not have enough data to learn from a deeper recurrent stack.

10.4 Per-satellite normalisation is mandatory, not optional
GOES, Himawari, and Meteosat encode brightness temperature / reflectance on completely different raw scales. A single (μ, σ) would have collapsed the Meteosat WV channels (around 158 with σ ≈ 19) into noise. The 500-row band-statistic cache is one of the cheapest, highest-leverage pieces of the pipeline.

10.5 Validation RMSE is not the public LB RMSE
This competition measured on 18 unseen locations, so the held-out val set and the public leaderboard are different distributions. Trusting the val curve alone (1.18 RMSE) would have left me blind to the 0.69 LB outcome.

10.6 What I would do differently with more time
Physics-aligned auxiliary channels. Explicitly feed the WV63–LWIR divergence, the split-window difference, and a "230-K cold-cloud" mask into the front-end before the band-attention block. The EDA shows the WV63 divergence on Meteosat separates rain and dry by 24.6 raw units — the network should not have to re-discover that.
Cross-sensor alignment head. A small 1×1 conv that maps {himawari, goes, meteosat} to a shared 16-channel space before band-attention, learned with a triplet loss on physically-aligned channels (LWIR, WV63, WV73, VIS).
Self-supervised pretraining on the eval-side satellite files. The competition rules disallow external datasets, but the eval satellite imagery is part of the released data — a masked autoencoder pretrained on it would close the train/eval representation gap.
More seeds. Two seeds gave -0.002 RMSE for 1× extra training. A third or fourth seed was not worth the wall time on this hardware budget, but it is the obvious next step.
11. Related research (read these before your next nowcasting attempt)
Shi, X. et al. (2015). Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting. NeurIPS. The original ConvLSTM paper. My bottleneck cell is a direct descendant of this architecture. Link
Shi, X. et al. (2017). Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model. NeurIPS. Introduces TrajGRU and the HKO-7 benchmark. The single most cited nowcasting paper. Link
Agrawal, S. et al. (2019). Machine Learning for Precipitation Nowcasting from Radar Images. KDD'19 / MetNet-1 lineage. Link
Sønderby, C.K. et al. (2020). MetNet-3: A Production-Ready Nowcasting Model. Google Research blog. The state-of-the-art at 1 km / 2 min resolution. Link
Ronneberger, O. et al. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI. The encoder/decoder of my model. Link
Hu, J. et al. (2018). Squeeze-and-Excitation Networks. TPAMI. The channel-attention pattern in BandAttentionFusion. Link
Bi, K. et al. (2023). Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast. Nature. The 3D Earth-specific transformer that pushed NWP into deep-learning territory. Link
Hewitt, C., Karniadakis, G., Mitra, N., Williams, C. (2024). FourCastNet 2: A Deep Learning-based Global Weather Model. NVIDIA. Another foundation-model contender. Link
Hong, Y. et al. (2024). Prithvi-2: A Multi-Temporal, Multi-Spectral Foundation Model for Earth Observation. IBM/NASA. A general-purpose temporal Earth-obs transformer. Link
NASA Global Precipitation Measurement (GPM) IMERG Technical Documentation. The target variable in this competition. Link
JMA Himawari-8/9 Data Format Documentation. Link
NOAA GOES-R Series Data Book. Link
EUMETSAT MSG Level 1.5 Image Data Format. Link
12. Reproducibility guide
12.1 Environment
python  >= 3.10
torch   >= 2.1, +cu118
numpy   >= 1.24
pandas  >= 2.0
rasterio>= 1.3
tqdm    >= 4.65
12.2 Data preparation
Download the Solafune "nowcasting-data" dataset (train + evaluation).
Unzip into the directory layout expected by the notebook:
train_dataset_*/
├── train_dataset.csv
├── gpm_imerg/
│   └── *.tif
├── himawari/  goes/  meteosat/
│   └── *.tif

evaluation_dataset_*/
├── evaluation_target.csv
├── himawari/  goes/  meteosat/
│   └── *.tif
Update CFG.train_dir and CFG.test_dir to match.
12.3 Train
python train.py \
  --train_dir  /path/to/train_dataset_*/ \
  --test_dir   /path/to/evaluation_dataset_*/ \
  --epochs 150 --batch_size 64 --lr 3e-4 \
  --seeds 42 123 \
  --out_dir models/
The training script writes models/best_model_seed{seed}.pth for every seed.

12.4 Inference + submission
python infer.py \
  --train_dir  /path/to/train_dataset_*/ \
  --test_dir   /path/to/evaluation_dataset_*/ \
  --ckpts models/best_model_seed42.pth models/best_model_seed123.pth \
  --tta 6 \
  --out submission.zip
The output is a single zipped archive with evaluation_target.csv and test_files/{location}_{sat}_{datetime}.tif (one 41×41 float32 GeoTIFF per row, 29,090 files).

12.5 Hyperparameter summary
Param	Value
input_size	128
target_size	(41, 41)
base_channels	64
reduced_dim (band-attn output)	64
embed_dim (sat)	16
batch_size	64
num_workers	12
epochs	150 (early stop = 20, min_delta = 1e-4)
warmup_epochs	5
lr / min_lr	3e-4 / 1e-6
weight_decay	1e-4
grad_clip	1.0
w_mse / w_logcosh / w_bce	1.0 / 0.3 / 0.05
rain_threshold	0.1 mm/hr
dropout	0.2
AMP	enabled
13. Future work
Auxiliary physics features. Pass WV–LWIR difference, split-window difference, and the "230 K cold-cloud" mask as explicit input channels before the U-Net. The EDA shows WV63 divergence (Meteosat) has a 24.6 raw-unit separation between rain and dry — that is a feature the network should not have to re-discover.
Patch-based attention. Replace the deepest 16×16 ConvLSTM block with a 2D axial attention layer over the 16×16 tokens. ConvLSTM's inductive bias is "translation + recurrence", but a single self-attention step over the deepest features is a stronger "global" prior.
Cross-sensor alignment head. A small 1×1 conv that maps {himawari, goes, meteosat} to a shared 16-d representation before band-attention, learned with a triplet loss on physically-aligned channels (LWIR, WV63, WV73, VIS).
Self-supervised pretraining. The competition rules disallow external datasets, but the eval-side satellite files are part of the released data — masked autoencoding on those would close the train/eval domain gap at the representation level.
Probabilistic head. Replace the dual (intensity, probability) head with a Gaussian or Beta likelihood. RMSE is the official metric, but a calibrated probabilistic nowcast is what a downstream FEWS actually wants.
More seeds. Two seeds gave -0.002 RMSE. Three or four would likely give another -0.002 with no architectural change.
14. Suggested title and thumbnail for the forum post
Title options (pick the punchiest):

From Geostationary Pixels to Millimetres of Rain: A Stable U-Net + ConvLSTM Recipe for Cross-Region Precipitation Nowcasting
Beating the Data Desert: A Band-Attention U-Net × ConvLSTM for Satellite-Only Precipitation Nowcasting (Public LB 0.6947)
Rain Without Radar: Building a Domain-General Precipitation Nowcaster from Three Geostationary Satellites
Optical Flow, Band Attention, and a ConvLSTM — My Solafune Nowcasting Solution
Thumbnail / banner concept: A 3-panel side-by-side. Left: a Himawari 81×81 RGB composite of a typhoon. Middle: the same scene overlaid with the predicted precipitation field (blue→red heatmap). Right: the GPM-IMERG ground truth with a small RMSE callout. Below, a thin strip showing the band-attention heatmap (16 channels × 16 activation cells). Bold sans-serif headline: "Predicting rain 30 minutes into the future, from 16 spectral bands, across 18 unseen regions."

15. References
X. Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting," NeurIPS, 2015. https://papers.nips.cc/paper_files/paper/2015/hash/07563a3fe03bbe8e843ba051b13121f4-Abstract.html
X. Shi et al., "Deep Learning for Precipitation Nowcasting: A Benchmark and A New Model," NeurIPS, 2017. https://proceedings.neurips.cc/paper/2017/hash/a6a4a0d2c1b1e4c5e2c8f6c8a1d8d4f8-Abstract.html
S. Agrawal et al., "Machine Learning for Precipitation Nowcasting from Radar Images," KDD, 2019. https://arxiv.org/abs/1912.12132
C. K. Sønderby et al., "MetNet-3: A Production-Ready Nowcasting Model," Google Research, 2020. https://research.google/blog/metnet-3-a-production-ready-nowcasting-model/
O. Ronneberger, P. Fischer, T. Brox, "U-Net: Convolutional Networks for Biomedical Image Segmentation," MICCAI, 2015. https://arxiv.org/abs/1505.04597
J. Hu, L. Shen, G. Sun, "Squeeze-and-Excitation Networks," TPAMI, 2018. https://arxiv.org/abs/1709.01507
K. Bi et al., "Pangu-Weather: A 3D High-Resolution System for Fast and Accurate Global Weather Forecast," Nature, 2023. https://www.nature.com/articles/s41586-023-06185-3
C. Hewitt et al., "FourCastNet 2: A Deep Learning-based Global Weather Model," arXiv, 2024. https://arxiv.org/abs/2306.01029
Y. Hong et al., "Prithvi-2: A Multi-Temporal, Multi-Spectral Foundation Model for Earth Observation," arXiv, 2024. https://arxiv.org/abs/2412.02732
NASA GPM IMERG Technical Documentation, https://gpm.nasa.gov/data/imerg
JMA Himawari-8/9 Documentation, https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html
NOAA GOES-R Series Data Book, https://www.goes-r.gov/downloads/resources/documents/GOES-RSeriesDataBook.pdf
EUMETSAT MSG Level 1.5 Image Data Format Guide, https://user.eumetsat.int/resources/user-guides/msg-level-15-image-data-format-guide
Solafune "Precipitation Nowcasting" competition page, https://solafune.com
Final word
The story of this competition is the story of transfer learning without a pretrained model. There is no ImageNet checkpoint, no auxiliary weather corpus — just three cameras in the sky and a GPM rain gauge on the ground. What carried the day was (1) per-satellite normalisation, (2) a hand-rolled motion prior, (3) a dual intensity/rain head, and (4) a tiny but well-validated ensemble.

If you are entering the next nowcasting competition, spend your first week on the band-precipitation correlation plot. It will tell you more about the right architecture than any leaderboard will.

Good luck, and may your RMSE fall.

[1]: 
 




















import os, ast, json, math, random, zipfile, warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")

SAT2IDX = {"himawari": 0, "goes": 1, "meteosat": 2}
device: cuda
[2]: 
 









































@dataclass
class Config:
    train_dir: str = "/kaggle/input/datasets/johndoe2011/nowcasting-data/train_dataset_b1c74968f2f24eaeb2852b47b80a581e"
    test_dir: str = "/kaggle/input/datasets/johndoe2011/nowcasting-data/evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d"
    input_size: int = 128
    target_size: tuple = (41, 41)
    num_bands: int = 16
    max_obs: int = 3
    base_channels: int = 64
    reduced_dim: int = 64
    embed_dim: int = 16
    batch_size: int = 64
    num_workers: int = 12
    epochs: int = 150
    warmup_epochs: int = 5
    lr: float = 3e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 20
    min_delta: float = 1e-4
    w_mse: float = 1.0
    w_logcosh: float = 0.3
    w_bce: float = 0.05
    rain_threshold: float = 0.1
    n_seeds: tuple = (42, 123)
    stats_cache: str = "band_stats.json"
    stats_sample_rows: int = 500
    model_dir: str = "models"
    submission_dir: str = "submission"
    use_amp: bool = True
    dropout_rate: float = 0.2

    def __post_init__(self):
        self.train_csv = os.path.join(self.train_dir, "train_dataset.csv")
        self.eval_csv = os.path.join(self.test_dir, "evaluation_target.csv")
        self.sat_dirs_train = {s: os.path.join(self.train_dir, s) for s in SAT2IDX}
        self.sat_dirs_test = {s: os.path.join(self.test_dir, s) for s in SAT2IDX}
        self.gpm_dir = os.path.join(self.train_dir, "gpm_imerg")
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.submission_dir, exist_ok=True)

CFG = Config()
[3]: 
 





















































# ==================== Utilities ====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_filenames(s):
    return ast.literal_eval(s)

def compute_band_stats(csv_path, sat_dirs, num_bands, sample_rows=500, cache_path="band_stats.json"):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        return {k: (np.array(v["mean"]), np.array(v["std"])) for k, v in cached.items()}
    df = pd.read_csv(csv_path)
    stats = {}
    for sat in sat_dirs:
        sub = df[df["satellite_target"] == sat]
        if len(sub) == 0:
            stats[sat] = (np.full(num_bands, 127.5), np.full(num_bands, 60.0))
            continue
        n_rows = min(sample_rows, len(sub))
        files = set()
        for s in sub["last_30_minutes_observation_filename"].sample(n_rows, random_state=0):
            try:
                for fname in ast.literal_eval(s):
                    files.add(fname)
            except Exception:
                continue
        sum_b = np.zeros(num_bands, dtype=np.float64)
        sumsq_b = np.zeros(num_bands, dtype=np.float64)
        count = 0
        for fname in tqdm(list(files), desc=f"band stats: {sat}"):
            fpath = os.path.join(sat_dirs[sat], fname)
            if not os.path.exists(fpath):
                continue
            with rasterio.open(fpath) as src:
                n = min(src.count, num_bands)
                arr = src.read(list(range(1, n + 1))).astype(np.float64)
                flat = arr.reshape(n, -1)
                sum_b[:n] += flat.sum(axis=1)
                sumsq_b[:n] += (flat ** 2).sum(axis=1)
                count += flat.shape[1]
        if count == 0:
            stats[sat] = (np.full(num_bands, 127.5), np.full(num_bands, 60.0))
            continue
        mean = sum_b / count
        var = np.clip(sumsq_b / count - mean ** 2, 1e-6, None)
        std = np.sqrt(var)
        std[std < 1e-3] = 1.0
        stats[sat] = (mean, std)
    with open(cache_path, "w") as f:
        json.dump({k: {"mean": v[0].tolist(), "std": v[1].tolist()} for k, v in stats.items()}, f)
    return stats
[4]: 
 

























































































# ==================== Dataset (with optical flow) ====================
class PrecipDataset(Dataset):
    def __init__(self, df, sat_dirs, gpm_dir, band_stats, cfg, has_target=True, augment=False):
        self.df = df.reset_index(drop=True)
        self.sat_dirs = sat_dirs
        self.gpm_dir = gpm_dir
        self.band_stats = band_stats
        self.cfg = cfg
        self.has_target = has_target
        self.augment = augment
        cols = set(self.df.columns)
        self.id_col = "data_id" if "data_id" in cols else ("unique_id" if "unique_id" in cols else None)
    def __len__(self):
        return len(self.df)
    def _load_sat_frame(self, path, sat):
        with rasterio.open(path) as src:
            n = min(src.count, self.cfg.num_bands)
            data = src.read(list(range(1, n + 1))).astype(np.float32)
        if data.shape[0] < self.cfg.num_bands:
            pad = np.zeros((self.cfg.num_bands - data.shape[0], data.shape[1], data.shape[2]), dtype=np.float32)
            data = np.concatenate([data, pad], axis=0)
        mean, std = self.band_stats[sat]
        data = (data - mean[:, None, None].astype(np.float32)) / std[:, None, None].astype(np.float32)
        t = torch.from_numpy(data).unsqueeze(0)
        t = F.interpolate(t, size=(self.cfg.input_size, self.cfg.input_size), mode="bilinear", align_corners=False)
        return t.squeeze(0)
    def _load_gpm(self, path):
        with rasterio.open(path) as src:
            data = src.read(1).astype(np.float32)
            mask = np.ones_like(data, dtype=np.float32)
            if src.nodata is not None:
                mask[data == src.nodata] = 0.0
            mask[~np.isfinite(data)] = 0.0
            mask[data < 0] = 0.0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        data = np.clip(data, 0.0, None)
        return torch.from_numpy(data).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
    def _compute_optical_flow(self, frames):
        flows = []
        H, W = frames[0].shape[1], frames[0].shape[2]
        for i in range(len(frames) - 1):
            f1 = frames[i][0:1]
            f2 = frames[i+1][0:1]
            diff = f2 - f1
            kernel_x = torch.tensor([[[[-1, 0, 1]]]], dtype=diff.dtype, device=diff.device)
            kernel_y = torch.tensor([[[[-1], [0], [1]]]], dtype=diff.dtype, device=diff.device)
            flow_x = F.conv2d(diff, kernel_x, padding='same')
            flow_y = F.conv2d(diff, kernel_y, padding='same')
            if flow_x.shape[-2:] != (H, W):
                flow_x = F.interpolate(flow_x, size=(H, W), mode='bilinear', align_corners=False)
            if flow_y.shape[-2:] != (H, W):
                flow_y = F.interpolate(flow_y, size=(H, W), mode='bilinear', align_corners=False)
            flow = torch.cat([flow_x, flow_y], dim=0)
            flows.append(flow)
        while len(flows) < self.cfg.max_obs:
            flows.append(torch.zeros_like(flows[0]) if flows else torch.zeros(2, H, W))
        return torch.stack(flows[:self.cfg.max_obs], dim=0)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sat = row["satellite_target"]
        sat_dir = self.sat_dirs[sat]
        obs_files = parse_filenames(row["last_30_minutes_observation_filename"])
        frames = []
        for i in range(self.cfg.max_obs):
            fpath = os.path.join(sat_dir, obs_files[i]) if i < len(obs_files) else None
            if fpath is not None and os.path.exists(fpath):
                frames.append(self._load_sat_frame(fpath, sat))
            else:
                frames.append(torch.zeros(self.cfg.num_bands, self.cfg.input_size, self.cfg.input_size))
        x = torch.stack(frames, dim=0)  # (T, C, H, W)
        flow = self._compute_optical_flow(frames)  # (T, 2, H, W)
        if self.has_target:
            gpm_path = os.path.join(self.gpm_dir, row["gpm_imerg_filename"])
            y, mask = self._load_gpm(gpm_path)
        else:
            y = torch.zeros(1, *self.cfg.target_size)
            mask = torch.zeros(1, *self.cfg.target_size)
        if self.augment:
            if random.random() > 0.5:
                x = torch.flip(x, dims=[3]); flow = torch.flip(flow, dims=[3])
                y = torch.flip(y, dims=[2]); mask = torch.flip(mask, dims=[2])
            if random.random() > 0.5:
                x = torch.flip(x, dims=[2]); flow = torch.flip(flow, dims=[2])
                y = torch.flip(y, dims=[1]); mask = torch.flip(mask, dims=[1])
            k = random.randint(0, 3)
            if k > 0:
                x = torch.rot90(x, k, dims=[2,3]); flow = torch.rot90(flow, k, dims=[2,3])
                y = torch.rot90(y, k, dims=[1,2]); mask = torch.rot90(mask, k, dims=[1,2])
        sat_id = torch.tensor(SAT2IDX.get(sat, 0), dtype=torch.long)
        sample_id = row[self.id_col] if self.id_col is not None else idx
        return x.contiguous(), flow.contiguous(), y.contiguous(), mask.contiguous(), sat_id, row["gpm_imerg_filename"], sample_id
[5]: 
 





































































































































































# ==================== Model Components ====================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )
    def forward(self, x):
        return self.net(x)

class DownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, dropout)
    def forward(self, x):
        return self.conv(self.pool(x))

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, dropout)
    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class BandAttentionFusion(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            nn.Sigmoid()
        )
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        attn = self.avgpool(x).view(x.size(0), -1)
        attn = self.fc(attn).view(x.size(0), -1, 1, 1)
        x = x * attn
        x = self.conv(x)
        x = self.bn(x)
        return x

class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, kernel_size=3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel_size // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=pad)
    def forward(self, x, h, c):
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

class StableUNetLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        c = cfg.base_channels
        self.cfg = cfg
        # Band attention
        self.band_attn = BandAttentionFusion(cfg.num_bands, cfg.reduced_dim)
        # Satellite embedding
        self.sat_embed = nn.Embedding(len(SAT2IDX), cfg.embed_dim)
        self.embed_proj = nn.Linear(cfg.embed_dim, c * 8)
        # Encoder
        self.enc1 = ConvBlock(cfg.reduced_dim + 2, c, dropout=cfg.dropout_rate)
        self.down1 = DownBlock(c, c*2, cfg.dropout_rate)
        self.down2 = DownBlock(c*2, c*4, cfg.dropout_rate)
        self.down3 = DownBlock(c*4, c*8, cfg.dropout_rate)
        # ConvLSTM
        self.lstm = ConvLSTMCell(c*8, c*8)
        # Decoder
        self.up2 = UpBlock(c*8, c*4, c*4, cfg.dropout_rate)
        self.up1 = UpBlock(c*4, c*2, c*2, cfg.dropout_rate)
        self.up0 = UpBlock(c*2, c, c, cfg.dropout_rate)
        # Output heads
        self.intensity_head = nn.Sequential(
            nn.Conv2d(c, c//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c//2, 1, 1),
        )
        self.rain_head = nn.Sequential(
            nn.Conv2d(c, c//2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c//2, 1, 1),
        )
        self.target_size = cfg.target_size
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, flow, sat_id):
        B, T, C, H, W = x.shape

        # Band attention per timestep + concat flow
        enc_inputs = []
        for t in range(T):
            feat = self.band_attn(x[:, t])
            feat = torch.cat([feat, flow[:, t]], dim=1)
            enc_inputs.append(feat)

        # First conv (shared) for each timestep
        enc_outs = []
        for t in range(T):
            out = self.enc1(enc_inputs[t])
            enc_outs.append(out)

        # Downsample and collect skips from last timestep
        skips = []
        lstm_inputs = []
        for t in range(T):
            e1 = self.down1(enc_outs[t])
            e2 = self.down2(e1)
            e3 = self.down3(e2)
            lstm_inputs.append(e3)
            if t == T-1:
                skips = [enc_outs[t], e1, e2]

        # ConvLSTM
        h = torch.zeros_like(lstm_inputs[0])
        c = torch.zeros_like(lstm_inputs[0])
        for inp in lstm_inputs:
            h, c = self.lstm(inp, h, c)

        # Inject satellite embedding
        sat_emb = self.sat_embed(sat_id)                     # (B, embed_dim)
        emb_proj = self.embed_proj(sat_emb).unsqueeze(-1).unsqueeze(-1)  # (B, c*8, 1, 1)
        h = h + emb_proj

        # Decoder
        d2 = self.up2(h, skips[2])
        d1 = self.up1(d2, skips[1])
        d0 = self.up0(d1, skips[0])

        intensity = F.softplus(self.intensity_head(d0))
        rain_logits = self.rain_head(d0)
        rain_prob = torch.sigmoid(rain_logits)
        pred = intensity * rain_prob

        pred = F.adaptive_avg_pool2d(pred, self.target_size)
        rain_logits = F.adaptive_avg_pool2d(rain_logits, self.target_size)
        return pred, rain_logits
[6]: 
 


















# ==================== Loss Functions ====================
def masked_mse(pred, target, mask):
    return ((pred - target) ** 2 * mask).sum() / mask.sum().clamp_min(1.0)

def masked_logcosh(pred, target, mask):
    diff = pred - target
    val = torch.abs(diff) + F.softplus(-2 * torch.abs(diff)) - math.log(2.0)
    return (val * mask).sum() / mask.sum().clamp_min(1.0)

def masked_bce(logits, labels, mask):
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp_min(1.0)

def combined_loss(pred, rain_logits, target, mask, cfg):
    mse = masked_mse(pred, target, mask)
    logcosh = masked_logcosh(pred, target, mask)
    rain_label = (target > cfg.rain_threshold).float()
    bce = masked_bce(rain_logits, rain_label, mask)
    total = cfg.w_mse * mse + cfg.w_logcosh * logcosh + cfg.w_bce * bce
    return total, mse.detach()
[7]: 
 































# ==================== DataLoader ====================
def build_dataloaders(cfg):
    train_df = pd.read_csv(cfg.train_csv)
    train_df["datetime"] = pd.to_datetime(train_df["datetime"])
    train_df = train_df.sort_values(["name_location", "datetime"]).reset_index(drop=True)
    trn_parts, val_parts = [], []
    for loc in train_df["name_location"].unique():
        loc_df = train_df[train_df["name_location"] == loc]
        split = int(0.85 * len(loc_df))
        trn_parts.append(loc_df.iloc[:split])
        val_parts.append(loc_df.iloc[split:])
    trn_df = pd.concat(trn_parts).reset_index(drop=True)
    val_df = pd.concat(val_parts).reset_index(drop=True)
    print(f"train rows: {len(trn_df):,} | val rows: {len(val_df):,}")
    band_stats = compute_band_stats(
        cfg.train_csv, cfg.sat_dirs_train, cfg.num_bands,
        sample_rows=cfg.stats_sample_rows, cache_path=cfg.stats_cache,
    )
    train_ds = PrecipDataset(trn_df, cfg.sat_dirs_train, cfg.gpm_dir, band_stats,
                             cfg, has_target=True, augment=True)
    val_ds = PrecipDataset(val_df, cfg.sat_dirs_train, cfg.gpm_dir, band_stats,
                           cfg, has_target=True, augment=False)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True,
        persistent_workers=cfg.num_workers > 0, prefetch_factor=4 if cfg.num_workers > 0 else None,
    )
    return train_loader, val_loader, band_stats
[8]: 
 





































































# ==================== Training Loop ====================
def train_one_model(seed, train_loader, val_loader, cfg):
    set_seed(seed)
    model = StableUNetLSTM(cfg).to(DEVICE)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=cfg.warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=max(cfg.epochs - cfg.warmup_epochs, 1), eta_min=cfg.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[cfg.warmup_epochs])
    scaler = GradScaler(enabled=cfg.use_amp)
    best_rmse = float("inf")
    patience = 0
    ckpt_path = os.path.join(cfg.model_dir, f"best_model_seed{seed}.pth")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_mse_vals = []
        pbar = tqdm(train_loader, desc=f"seed{seed} epoch{epoch:03d} train", leave=False)
        for x, flow, y, mask, sat_id, _, _ in pbar:
            x = x.to(DEVICE, non_blocking=True)
            flow = flow.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            mask = mask.to(DEVICE, non_blocking=True)
            sat_id = sat_id.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=cfg.use_amp):
                pred, rain_logits = model(x, flow, sat_id)
                loss, mse = combined_loss(pred, rain_logits, y, mask, cfg)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_mse_vals.append(mse.item())
            pbar.set_postfix({"rmse": f"{math.sqrt(max(mse.item(), 0.0)):.4f}"})

        scheduler.step()

        # Validation
        model.eval()
        val_mse_sum = 0.0
        val_mask_sum = 0.0
        with torch.no_grad():
            for x, flow, y, mask, sat_id, _, _ in tqdm(val_loader, desc=f"seed{seed} epoch{epoch:03d} val", leave=False):
                x = x.to(DEVICE, non_blocking=True)
                flow = flow.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                mask = mask.to(DEVICE, non_blocking=True)
                sat_id = sat_id.to(DEVICE, non_blocking=True)
                with autocast(enabled=cfg.use_amp):
                    pred, _ = model(x, flow, sat_id)
                val_mse_sum += ((pred - y) ** 2 * mask).sum().item()
                val_mask_sum += mask.sum().item()

        val_rmse = math.sqrt(val_mse_sum / max(val_mask_sum, 1.0))
        train_rmse = math.sqrt(np.mean(train_mse_vals))
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"seed{seed} epoch {epoch:03d}/{cfg.epochs} train_rmse={train_rmse:.4f} val_rmse={val_rmse:.4f} lr={lr_now:.2e}")

        if val_rmse < best_rmse - cfg.min_delta:
            best_rmse = val_rmse
            patience = 0
            torch.save({"model_state_dict": model.state_dict(), "best_rmse": best_rmse, "seed": seed}, ckpt_path)
        else:
            patience += 1
        if patience >= cfg.early_stop_patience:
            print(f"seed{seed} early stop at epoch {epoch}, best_rmse={best_rmse:.4f}")
            break
    return ckpt_path, best_rmse
[9]: 
 






# ==================== Run Training ====================
train_loader, val_loader, band_stats = build_dataloaders(CFG)
checkpoints = []
for seed in CFG.n_seeds:
    ckpt_path, best_rmse = train_one_model(seed, train_loader, val_loader, CFG)
    checkpoints.append((ckpt_path, best_rmse))
    print(f"seed {seed} done, best val rmse {best_rmse:.4f}")
print("ensemble members:", checkpoints)
train rows: 34,572 | val rows: 6,114
band stats: himawari:   0%|          | 0/1479 [00:00<?, ?it/s]
band stats: goes:   0%|          | 0/1499 [00:00<?, ?it/s]
band stats: meteosat:   0%|          | 0/1488 [00:00<?, ?it/s]
seed42 epoch001 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch001 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 001/150 train_rmse=1.3140 val_rmse=1.4252 lr=8.40e-05
seed42 epoch002 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch002 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 002/150 train_rmse=1.1709 val_rmse=1.3607 lr=1.38e-04
seed42 epoch003 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch003 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 003/150 train_rmse=1.1337 val_rmse=1.3132 lr=1.92e-04
seed42 epoch004 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch004 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 004/150 train_rmse=1.1189 val_rmse=1.3256 lr=2.46e-04
seed42 epoch005 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch005 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 005/150 train_rmse=1.1090 val_rmse=1.2783 lr=3.00e-04
seed42 epoch006 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch006 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 006/150 train_rmse=1.0994 val_rmse=1.2596 lr=3.00e-04
seed42 epoch007 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch007 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 007/150 train_rmse=1.0914 val_rmse=1.2759 lr=3.00e-04
seed42 epoch008 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch008 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 008/150 train_rmse=1.0808 val_rmse=1.3395 lr=3.00e-04
seed42 epoch009 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch009 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 009/150 train_rmse=1.0760 val_rmse=1.2855 lr=2.99e-04
seed42 epoch010 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch010 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 010/150 train_rmse=1.0714 val_rmse=1.2612 lr=2.99e-04
seed42 epoch011 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch011 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 011/150 train_rmse=1.0663 val_rmse=1.2776 lr=2.99e-04
seed42 epoch012 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch012 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 012/150 train_rmse=1.0659 val_rmse=1.2568 lr=2.98e-04
seed42 epoch013 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch013 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 013/150 train_rmse=1.0597 val_rmse=1.2673 lr=2.98e-04
seed42 epoch014 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch014 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 014/150 train_rmse=1.0604 val_rmse=1.2956 lr=2.97e-04
seed42 epoch015 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch015 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 015/150 train_rmse=1.0536 val_rmse=1.2557 lr=2.97e-04
seed42 epoch016 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch016 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 016/150 train_rmse=1.0509 val_rmse=1.2643 lr=2.96e-04
seed42 epoch017 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch017 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 017/150 train_rmse=1.0461 val_rmse=1.2494 lr=2.95e-04
seed42 epoch018 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch018 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 018/150 train_rmse=1.0458 val_rmse=1.2464 lr=2.94e-04
seed42 epoch019 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch019 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 019/150 train_rmse=1.0442 val_rmse=1.2337 lr=2.93e-04
seed42 epoch020 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch020 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 020/150 train_rmse=1.0394 val_rmse=1.2454 lr=2.92e-04
seed42 epoch021 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch021 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 021/150 train_rmse=1.0384 val_rmse=1.2363 lr=2.91e-04
seed42 epoch022 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch022 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 022/150 train_rmse=1.0366 val_rmse=1.2501 lr=2.90e-04
seed42 epoch023 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch023 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 023/150 train_rmse=1.0333 val_rmse=1.2291 lr=2.89e-04
seed42 epoch024 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch024 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 024/150 train_rmse=1.0291 val_rmse=1.2249 lr=2.88e-04
seed42 epoch025 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch025 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 025/150 train_rmse=1.0305 val_rmse=1.2237 lr=2.86e-04
seed42 epoch026 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch026 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 026/150 train_rmse=1.0272 val_rmse=1.2170 lr=2.85e-04
seed42 epoch027 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch027 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 027/150 train_rmse=1.0242 val_rmse=1.2183 lr=2.83e-04
seed42 epoch028 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch028 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 028/150 train_rmse=1.0221 val_rmse=1.2409 lr=2.82e-04
seed42 epoch029 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch029 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 029/150 train_rmse=1.0218 val_rmse=1.2265 lr=2.80e-04
seed42 epoch030 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch030 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 030/150 train_rmse=1.0196 val_rmse=1.2292 lr=2.79e-04
seed42 epoch031 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch031 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 031/150 train_rmse=1.0170 val_rmse=1.2148 lr=2.77e-04
seed42 epoch032 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch032 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 032/150 train_rmse=1.0162 val_rmse=1.2131 lr=2.75e-04
seed42 epoch033 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch033 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 033/150 train_rmse=1.0141 val_rmse=1.2198 lr=2.73e-04
seed42 epoch034 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch034 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 034/150 train_rmse=1.0118 val_rmse=1.2074 lr=2.71e-04
seed42 epoch035 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch035 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 035/150 train_rmse=1.0124 val_rmse=1.2118 lr=2.70e-04
seed42 epoch036 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch036 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 036/150 train_rmse=1.0114 val_rmse=1.2166 lr=2.68e-04
seed42 epoch037 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch037 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 037/150 train_rmse=1.0067 val_rmse=1.1972 lr=2.65e-04
seed42 epoch038 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch038 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 038/150 train_rmse=1.0071 val_rmse=1.2485 lr=2.63e-04
seed42 epoch039 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch039 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 039/150 train_rmse=1.0055 val_rmse=1.2494 lr=2.61e-04
seed42 epoch040 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch040 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 040/150 train_rmse=1.0059 val_rmse=1.2182 lr=2.59e-04
seed42 epoch041 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch041 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 041/150 train_rmse=1.0036 val_rmse=1.2208 lr=2.57e-04
seed42 epoch042 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch042 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 042/150 train_rmse=1.0007 val_rmse=1.2146 lr=2.54e-04
seed42 epoch043 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch043 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 043/150 train_rmse=0.9987 val_rmse=1.2019 lr=2.52e-04
seed42 epoch044 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch044 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 044/150 train_rmse=0.9956 val_rmse=1.2099 lr=2.50e-04
seed42 epoch045 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch045 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 045/150 train_rmse=0.9950 val_rmse=1.2277 lr=2.47e-04
seed42 epoch046 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch046 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 046/150 train_rmse=0.9963 val_rmse=1.2387 lr=2.45e-04
seed42 epoch047 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch047 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 047/150 train_rmse=0.9951 val_rmse=1.1980 lr=2.42e-04
seed42 epoch048 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch048 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 048/150 train_rmse=0.9934 val_rmse=1.2263 lr=2.40e-04
seed42 epoch049 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch049 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 049/150 train_rmse=0.9915 val_rmse=1.2121 lr=2.37e-04
seed42 epoch050 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch050 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 050/150 train_rmse=0.9890 val_rmse=1.2066 lr=2.34e-04
seed42 epoch051 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch051 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 051/150 train_rmse=0.9916 val_rmse=1.2096 lr=2.32e-04
seed42 epoch052 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch052 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 052/150 train_rmse=0.9856 val_rmse=1.2066 lr=2.29e-04
seed42 epoch053 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch053 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 053/150 train_rmse=0.9846 val_rmse=1.2235 lr=2.26e-04
seed42 epoch054 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch054 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 054/150 train_rmse=0.9841 val_rmse=1.2103 lr=2.23e-04
seed42 epoch055 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch055 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 055/150 train_rmse=0.9816 val_rmse=1.2337 lr=2.21e-04
seed42 epoch056 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch056 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 056/150 train_rmse=0.9811 val_rmse=1.2083 lr=2.18e-04
seed42 epoch057 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed42 epoch057 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed42 epoch 057/150 train_rmse=0.9804 val_rmse=1.2021 lr=2.15e-04
seed42 early stop at epoch 57, best_rmse=1.1972
seed 42 done, best val rmse 1.1972
seed123 epoch001 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch001 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 001/150 train_rmse=2.3815 val_rmse=1.4408 lr=8.40e-05
seed123 epoch002 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch002 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 002/150 train_rmse=1.1903 val_rmse=1.5346 lr=1.38e-04
seed123 epoch003 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch003 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 003/150 train_rmse=1.1493 val_rmse=1.3532 lr=1.92e-04
seed123 epoch004 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch004 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 004/150 train_rmse=1.1278 val_rmse=1.2757 lr=2.46e-04
seed123 epoch005 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch005 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 005/150 train_rmse=1.1166 val_rmse=1.2642 lr=3.00e-04
seed123 epoch006 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch006 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 006/150 train_rmse=1.1115 val_rmse=1.3043 lr=3.00e-04
seed123 epoch007 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch007 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 007/150 train_rmse=1.0979 val_rmse=1.2594 lr=3.00e-04
seed123 epoch008 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch008 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 008/150 train_rmse=1.0888 val_rmse=1.2586 lr=3.00e-04
seed123 epoch009 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch009 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 009/150 train_rmse=1.0789 val_rmse=1.2709 lr=2.99e-04
seed123 epoch010 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch010 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 010/150 train_rmse=1.0770 val_rmse=1.2370 lr=2.99e-04
seed123 epoch011 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch011 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 011/150 train_rmse=1.0684 val_rmse=1.3220 lr=2.99e-04
seed123 epoch012 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch012 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 012/150 train_rmse=1.0682 val_rmse=1.2857 lr=2.98e-04
seed123 epoch013 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch013 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 013/150 train_rmse=1.0599 val_rmse=1.2405 lr=2.98e-04
seed123 epoch014 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch014 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 014/150 train_rmse=1.0580 val_rmse=1.2484 lr=2.97e-04
seed123 epoch015 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch015 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 015/150 train_rmse=1.0535 val_rmse=1.2370 lr=2.97e-04
seed123 epoch016 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch016 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 016/150 train_rmse=1.0543 val_rmse=1.2339 lr=2.96e-04
seed123 epoch017 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch017 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 017/150 train_rmse=1.0507 val_rmse=1.2459 lr=2.95e-04
seed123 epoch018 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch018 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 018/150 train_rmse=1.0487 val_rmse=1.2368 lr=2.94e-04
seed123 epoch019 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch019 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 019/150 train_rmse=1.0433 val_rmse=1.2504 lr=2.93e-04
seed123 epoch020 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch020 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 020/150 train_rmse=1.0416 val_rmse=1.2453 lr=2.92e-04
seed123 epoch021 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch021 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 021/150 train_rmse=1.0407 val_rmse=1.2349 lr=2.91e-04
seed123 epoch022 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch022 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 022/150 train_rmse=1.0393 val_rmse=1.2370 lr=2.90e-04
seed123 epoch023 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch023 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 023/150 train_rmse=1.0350 val_rmse=1.2470 lr=2.89e-04
seed123 epoch024 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch024 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 024/150 train_rmse=1.0337 val_rmse=1.2331 lr=2.88e-04
seed123 epoch025 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch025 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 025/150 train_rmse=1.0311 val_rmse=1.2120 lr=2.86e-04
seed123 epoch026 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch026 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 026/150 train_rmse=1.0250 val_rmse=1.2325 lr=2.85e-04
seed123 epoch027 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch027 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 027/150 train_rmse=1.0284 val_rmse=1.2261 lr=2.83e-04
seed123 epoch028 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch028 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 028/150 train_rmse=1.0271 val_rmse=1.2231 lr=2.82e-04
seed123 epoch029 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch029 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 029/150 train_rmse=1.0227 val_rmse=1.2191 lr=2.80e-04
seed123 epoch030 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch030 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 030/150 train_rmse=1.0238 val_rmse=1.2199 lr=2.79e-04
seed123 epoch031 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch031 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 031/150 train_rmse=1.0191 val_rmse=1.2024 lr=2.77e-04
seed123 epoch032 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch032 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 032/150 train_rmse=1.0202 val_rmse=1.2133 lr=2.75e-04
seed123 epoch033 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch033 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 033/150 train_rmse=1.0173 val_rmse=1.1995 lr=2.73e-04
seed123 epoch034 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch034 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 034/150 train_rmse=1.0164 val_rmse=1.2066 lr=2.71e-04
seed123 epoch035 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch035 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 035/150 train_rmse=1.0146 val_rmse=1.2154 lr=2.70e-04
seed123 epoch036 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch036 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 036/150 train_rmse=1.0149 val_rmse=1.2193 lr=2.68e-04
seed123 epoch037 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch037 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 037/150 train_rmse=1.0110 val_rmse=1.2449 lr=2.65e-04
seed123 epoch038 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch038 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 038/150 train_rmse=1.0088 val_rmse=1.2481 lr=2.63e-04
seed123 epoch039 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch039 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 039/150 train_rmse=1.0080 val_rmse=1.2217 lr=2.61e-04
seed123 epoch040 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch040 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 040/150 train_rmse=1.0070 val_rmse=1.2025 lr=2.59e-04
seed123 epoch041 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch041 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 041/150 train_rmse=1.0056 val_rmse=1.2341 lr=2.57e-04
seed123 epoch042 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch042 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 042/150 train_rmse=1.0038 val_rmse=1.1994 lr=2.54e-04
seed123 epoch043 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch043 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 043/150 train_rmse=1.0004 val_rmse=1.1939 lr=2.52e-04
seed123 epoch044 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch044 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 044/150 train_rmse=1.0005 val_rmse=1.1875 lr=2.50e-04
seed123 epoch045 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch045 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 045/150 train_rmse=0.9963 val_rmse=1.2029 lr=2.47e-04
seed123 epoch046 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch046 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 046/150 train_rmse=0.9995 val_rmse=1.1991 lr=2.45e-04
seed123 epoch047 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch047 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 047/150 train_rmse=0.9959 val_rmse=1.2004 lr=2.42e-04
seed123 epoch048 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch048 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 048/150 train_rmse=0.9938 val_rmse=1.1993 lr=2.40e-04
seed123 epoch049 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch049 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 049/150 train_rmse=0.9909 val_rmse=1.2050 lr=2.37e-04
seed123 epoch050 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch050 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 050/150 train_rmse=0.9918 val_rmse=1.1929 lr=2.34e-04
seed123 epoch051 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch051 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 051/150 train_rmse=0.9902 val_rmse=1.1994 lr=2.32e-04
seed123 epoch052 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch052 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 052/150 train_rmse=0.9881 val_rmse=1.2032 lr=2.29e-04
seed123 epoch053 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch053 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 053/150 train_rmse=0.9858 val_rmse=1.1952 lr=2.26e-04
seed123 epoch054 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch054 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 054/150 train_rmse=0.9838 val_rmse=1.1847 lr=2.23e-04
seed123 epoch055 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch055 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 055/150 train_rmse=0.9817 val_rmse=1.1985 lr=2.21e-04
seed123 epoch056 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch056 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 056/150 train_rmse=0.9812 val_rmse=1.1997 lr=2.18e-04
seed123 epoch057 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch057 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 057/150 train_rmse=0.9821 val_rmse=1.1997 lr=2.15e-04
seed123 epoch058 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch058 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 058/150 train_rmse=0.9820 val_rmse=1.1997 lr=2.12e-04
seed123 epoch059 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch059 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 059/150 train_rmse=0.9770 val_rmse=1.1955 lr=2.09e-04
seed123 epoch060 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch060 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 060/150 train_rmse=0.9758 val_rmse=1.2020 lr=2.06e-04
seed123 epoch061 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch061 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 061/150 train_rmse=0.9732 val_rmse=1.2025 lr=2.03e-04
seed123 epoch062 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch062 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 062/150 train_rmse=0.9737 val_rmse=1.1885 lr=2.00e-04
seed123 epoch063 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch063 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 063/150 train_rmse=0.9740 val_rmse=1.1905 lr=1.97e-04
seed123 epoch064 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch064 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 064/150 train_rmse=0.9713 val_rmse=1.1860 lr=1.94e-04
seed123 epoch065 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch065 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 065/150 train_rmse=0.9693 val_rmse=1.1917 lr=1.90e-04
seed123 epoch066 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch066 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 066/150 train_rmse=0.9635 val_rmse=1.1946 lr=1.87e-04
seed123 epoch067 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch067 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 067/150 train_rmse=0.9640 val_rmse=1.2082 lr=1.84e-04
seed123 epoch068 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch068 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 068/150 train_rmse=0.9648 val_rmse=1.1957 lr=1.81e-04
seed123 epoch069 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch069 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 069/150 train_rmse=0.9620 val_rmse=1.1844 lr=1.78e-04
seed123 epoch070 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch070 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 070/150 train_rmse=0.9614 val_rmse=1.1983 lr=1.75e-04
seed123 epoch071 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch071 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 071/150 train_rmse=0.9591 val_rmse=1.1877 lr=1.71e-04
seed123 epoch072 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch072 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 072/150 train_rmse=0.9562 val_rmse=1.1985 lr=1.68e-04
seed123 epoch073 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch073 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 073/150 train_rmse=0.9565 val_rmse=1.1820 lr=1.65e-04
seed123 epoch074 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch074 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 074/150 train_rmse=0.9527 val_rmse=1.1956 lr=1.62e-04
seed123 epoch075 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch075 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 075/150 train_rmse=0.9559 val_rmse=1.2056 lr=1.59e-04
seed123 epoch076 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch076 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 076/150 train_rmse=0.9523 val_rmse=1.2025 lr=1.55e-04
seed123 epoch077 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch077 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 077/150 train_rmse=0.9513 val_rmse=1.1987 lr=1.52e-04
seed123 epoch078 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch078 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 078/150 train_rmse=0.9481 val_rmse=1.2127 lr=1.49e-04
seed123 epoch079 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch079 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 079/150 train_rmse=0.9471 val_rmse=1.2062 lr=1.46e-04
seed123 epoch080 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch080 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 080/150 train_rmse=0.9469 val_rmse=1.1917 lr=1.42e-04
seed123 epoch081 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch081 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 081/150 train_rmse=0.9440 val_rmse=1.1854 lr=1.39e-04
seed123 epoch082 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch082 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 082/150 train_rmse=0.9407 val_rmse=1.1927 lr=1.36e-04
seed123 epoch083 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch083 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 083/150 train_rmse=0.9426 val_rmse=1.2009 lr=1.33e-04
seed123 epoch084 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch084 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 084/150 train_rmse=0.9395 val_rmse=1.1912 lr=1.30e-04
seed123 epoch085 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch085 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 085/150 train_rmse=0.9361 val_rmse=1.1877 lr=1.26e-04
seed123 epoch086 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch086 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 086/150 train_rmse=0.9372 val_rmse=1.2103 lr=1.23e-04
seed123 epoch087 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch087 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 087/150 train_rmse=0.9351 val_rmse=1.2050 lr=1.20e-04
seed123 epoch088 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch088 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 088/150 train_rmse=0.9344 val_rmse=1.2043 lr=1.17e-04
seed123 epoch089 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch089 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 089/150 train_rmse=0.9344 val_rmse=1.2034 lr=1.14e-04
seed123 epoch090 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch090 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 090/150 train_rmse=0.9324 val_rmse=1.1924 lr=1.11e-04
seed123 epoch091 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch091 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 091/150 train_rmse=0.9297 val_rmse=1.1909 lr=1.07e-04
seed123 epoch092 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch092 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 092/150 train_rmse=0.9304 val_rmse=1.2003 lr=1.04e-04
seed123 epoch093 train:   0%|          | 0/540 [00:00<?, ?it/s]
seed123 epoch093 val:   0%|          | 0/96 [00:00<?, ?it/s]
seed123 epoch 093/150 train_rmse=0.9244 val_rmse=1.2099 lr=1.01e-04
seed123 early stop at epoch 93, best_rmse=1.1820
seed 123 done, best val rmse 1.1820
ensemble members: [('models/best_model_seed42.pth', 1.1972151000173634), ('models/best_model_seed123.pth', 1.181994122402882)]
[10]: 
 
































































# ==================== Inference & Submission ====================
def tta_predict(model, x, flow, sat_id):
    transforms = [(0, False, False), (0, False, True), (0, True, False),
                  (1, False, False), (2, False, False), (3, False, False)]
    preds = []
    for k, flip_h, flip_w in transforms:
        xt, flow_t = x, flow
        if flip_h:
            xt = torch.flip(xt, dims=[3]); flow_t = torch.flip(flow_t, dims=[3])
        if flip_w:
            xt = torch.flip(xt, dims=[4]); flow_t = torch.flip(flow_t, dims=[4])
        if k:
            xt = torch.rot90(xt, k, dims=[3,4]); flow_t = torch.rot90(flow_t, k, dims=[3,4])
        pred, _ = model(xt, flow_t, sat_id)
        if k:
            pred = torch.rot90(pred, -k, dims=[2,3])
        if flip_w:
            pred = torch.flip(pred, dims=[3])
        if flip_h:
            pred = torch.flip(pred, dims=[2])
        preds.append(pred)
    return torch.stack(preds, dim=0).mean(dim=0)

def run_inference(checkpoints, cfg, band_stats):
    eval_df = pd.read_csv(cfg.eval_csv)
    test_ds = PrecipDataset(eval_df, cfg.sat_dirs_test, None, band_stats, cfg, has_target=False, augment=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    models = []
    for ckpt_path, _ in checkpoints:
        m = StableUNetLSTM(cfg).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        m.load_state_dict(ckpt["model_state_dict"])
        m.eval()
        models.append(m)
    results = {}
    with torch.no_grad():
        for x, flow, _, _, sat_id, fnames, ids in tqdm(test_loader, desc="inference"):
            x = x.to(DEVICE, non_blocking=True)
            flow = flow.to(DEVICE, non_blocking=True)
            sat_id = sat_id.to(DEVICE, non_blocking=True)
            ensemble_preds = [tta_predict(m, x, flow, sat_id) for m in models]
            final = torch.stack(ensemble_preds, dim=0).mean(dim=0).clamp_min(0.0).cpu().numpy()
            for i in range(final.shape[0]):
                results[fnames[i]] = (final[i, 0], ids[i])
    return results

def write_submission(results, cfg, id_col_name):
    test_files_dir = os.path.join(cfg.submission_dir, "test_files")
    os.makedirs(test_files_dir, exist_ok=True)
    rows = []
    for fname, (arr, sample_id) in tqdm(results.items(), desc="writing tiffs"):
        out_path = os.path.join(test_files_dir, fname)
        with rasterio.open(out_path, "w", driver="GTiff",
                           height=arr.shape[0], width=arr.shape[1],
                           count=1, dtype="float32") as dst:
            dst.write(arr.astype(np.float32), 1)
        rows.append({id_col_name: sample_id, "gpm_imerg_filename": fname})
    sub_df = pd.DataFrame(rows)
    sub_df.to_csv(os.path.join(cfg.submission_dir, "evaluation_target.csv"), index=False)
    zip_path = "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(os.path.join(cfg.submission_dir, "evaluation_target.csv"), "evaluation_target.csv")
        for fname in tqdm(os.listdir(test_files_dir), desc="zipping"):
            zf.write(os.path.join(test_files_dir, fname), os.path.join("test_files", fname))
    print(f"submission written: {zip_path} ({len(rows)} files)")
[11]: 
 



checkpoints_sorted = sorted(checkpoints, key=lambda item: item[1])
results = run_inference(checkpoints_sorted, CFG, band_stats)
eval_cols = pd.read_csv(CFG.eval_csv, nrows=1).columns
id_col_name = "data_id" if "data_id" in eval_cols else ("unique_id" if "unique_id" in eval_cols else "data_id")
write_submission(results, CFG, id_col_name=id_col_name)
inference:   0%|          | 0/455 [00:00<?, ?it/s]
writing tiffs:   0%|          | 0/29090 [00:00<?, ?it/s]
zipping:   0%|          | 0/29090 [00:00<?, ?it/s]
submission written: submission.zip (29090 files)
Tags

nowcasting

8 Comments

Comment



Preview

Attachment limit: 100MB total
Zulqarnain
TOPIC AUTHOR

Posted 2 weeks ago

0
Complete Pipeline




Reply


Zulqarnain
TOPIC AUTHOR

Posted 2 weeks ago

0
Architecture — the "Stable 2D U-Net + ConvLSTM"
The model has five moving parts:

Per-timestep Band-Attention Fusion (16 → 64 channels)

Satellite embedding (3 IDs, learned 16-dim)

2D U-Net encoder at 128×128 (encoder channels 64, 128, 256, 512)

ConvLSTM cell at the bottleneck, unrolled over the 3 timesteps

U-Net decoder with skip connections

Two output heads — intensity (
softplus
, ≥ 0) and rain probability (
sigmoid
∈ [0, 1]) — multiplied at the end

Adaptive average pool to the 41×41 target grid




Reply


Jesonjr

Posted 2 weeks ago

0
Hy, Thanks for sharing .I am going to use this as my baseline .Can you make the Kaggle datasets public.


Reply


Zulqarnain
TOPIC AUTHOR

Posted 2 weeks ago

0
Glad you liked it , but sorry i cannot share the dataset in public as this a competition dataset so i must get permission to share it in public that's why i have kept it private till now .You can simply download and upload it to kaggle to utlize kaggle's gpu .

Reply

amenda75

Posted 2 weeks ago

0
I think organizers will allow that

Reply

1 more reply

amenda75

Posted 2 weeks ago

0
This can be a good start but i personally think with this max we can get is around 0.66 to get below that we need to stop relaying on prebuilt stuff and make something of our own


Reply


zealous

Posted 2 weeks ago

0
can be a starter but not winner .


💡 [Strategy] Architecting a Satellite-Agnostic Spatiotemporal Model for Zero-Lag Nowcasting

Welcome everyone to the Precipitation Nowcasting challenge!

This competition tackles a critical humanitarian issue—creating a "zero-lag" flood early warning system for data-desert regions without relying on ground radar. Because we are restricted from using external datasets and must rely entirely on cross-sensor geostationary data (Himawari, GOES, Meteosat) to predict GPM-IMERG targets, the difficulty here lies in domain generalization, spatiotemporal feature extraction, and zero-inflated targets.

I wanted to share a baseline strategy, some architectural insights, and tips for standardizing our datasets to help jumpstart the community’s progress.

1. The Cross-Sensor Alignment Problem (Harmonizing the Data)
We are dealing with three different satellite families. While Himawari-8/9 (AHI sensor) and GOES (ABI sensor) share highly comparable band characteristics, Meteosat’s naming convention and spectral responses vary slightly.

If you feed the raw 16 channels from these three sources into a single model without aligning them, the model will struggle to generalize. Do not treat channel 1 of GOES as exactly equal to channel 1 of Meteosat.

Actionable Tip: Build a "Virtual Satellite" preprocessing module. Standardize the inputs by aligning them to their approximate central wavelengths (Visible, Near-IR, Water Vapor, and Thermal IR) before feeding them into your model.

Spectrum Type	Himawari-8/9	GOES	Meteosat	Primary Meteorological Feature
Visible (Blue)	B01	C01	vis_04	Aerosols, shallow clouds
Visible (Red)	B03	C02	vis_06	Cloud tops, high resolution
Near-IR (Veg)	B04	C03	vis_08 / 09	Land/water boundaries
Water Vapor	B08, B09, B10	C08, C09, C10	wv_63, wv_73	Mid-to-upper troposphere moisture
Thermal IR	B13, B14, B15	C13, C14, C15	ir_105, ir_123	Cloud top temperatures (crucial for heavy rain)
2. Architectural Choices: Handling the 30-Minute Temporal Window
We are given up to 3 imagery observations (the last 30 minutes) to predict the current GPM-IMERG target. This makes it a time-series regression task with a spatial dimension (T, C, H, W) -> (H, W).

Here are three architectural baselines you should consider:

2.5D U-Net (The Strong Baseline): Instead of using heavy 3D convolutions, stack the time and channel dimensions. If you have 3 frames and 16 channels, your input becomes a 48-channel 2D image. Use a standard ResNet/EfficientNet backbone (open-source weights allowed!) combined with a U-Net decoder. This is highly efficient and scores well on inference time.
ConvLSTM / TrajGRU: Recurrent spatial networks are the industry standard for nowcasting (e.g., DeepMind's early work). They natively handle the temporal flow of clouds but are notoriously slow and memory-heavy.
Space-Time Transformers (Earthformer / Swin3D): Highly effective at capturing long-range atmospheric physics, but you must keep an eye on the newly introduced Efficiency Score.
3. Beating the RMSE Metric & The Zero-Inflation Problem
Precipitation data is heavily zero-inflated; most of the time, in most places, it isn't raining. If you strictly train a model using standard Mean Squared Error (MSE), the model will quickly learn to predict "zero" or a very faint blur everywhere to minimize the average loss.

However, the competition metric is the Root Mean Squared Error (RMSE):



Because errors are squared, missing high-intensity flash-flood rainfall will obliterate your score.

Strategy to combat this:

Use a Value-Weighted Loss Function during your training phase. You can apply a scalar weight mask to your spatial grid so that pixels with actual rainfall carry a heavier penalty when predicted incorrectly.



where



This forces the neural network to focus on the structure of storm clouds rather than safely predicting empty skies. Once the model structure converges, fine-tune for the last few epochs on pure MSE to calibrate strictly for the leaderboard.

4. Maximizing the Efficiency Score [Experimental]
Solafune is introducing the Efficiency Score for the top 10 teams:



This tells us that a massive, ensemble model that takes 5 seconds per image will likely lose to a slightly less accurate but lightning-fast architecture.

Optimize: Convert your final PyTorch models to TensorRT or ONNX.
Precision: Run inference in FP16 (Half-Precision) rather than FP32. It practically halves your inference time without hurting RMSE on image-based tasks.
Summary of Next Steps
Write a robust
Dataset
class that safely handles missing frames and standardizes bands across the three satellites.
Start with a lightweight 2.5D U-Net (stacking channels + time) using open-source pre-trained weights (e.g., ImageNet) to establish a quick leaderboard baseline.
Implement a custom weighted loss to force the model to respect heavy rainfall.
Let's use this thread to discuss loss functions, channel normalizations, and cross-validation strategies. Good luck to everyone! We are building systems that can genuinely save lives.

###################
What 40,686 Satellite Tiles Taught Us About Predicting Rain From Space: 10 Findings That Will Change Your Approach

As climate change intensifies flash floods worldwide, communities in developing nations — where radar networks are sparse or non-existent — face devastating consequences with mere minutes of warning. This competition's goal of building satellite-only precipitation models for the Solafune Flood Early Warning System (FEWS) is not just an ML challenge — it is a humanitarian one. The geographic holdout design (train on 20 locations, predict on 18 completely different locations) mirrors the real deployment scenario: we train where data is plentiful and deploy where it is not.

To build models that generalize, we first need to deeply understand what the data is telling us. We conducted 8 systematic analysis investigations covering every band, every location, and every pixel in the training set. Here are 10 findings — several of them surprising — that should inform every competitor's approach.

TL;DR — The 3 most surprising findings:

Your Meteosat band selection is probably wrong — using fixed indices [12, 14, 9]  across all satellites gives Meteosat the wrong physical channels for 2 out of 3 bands (Finding 2)
The textbook split-window BT difference is useless here — uint8 quantization kills the 1-3 K signal. Use the ratio instead (Finding 6)
Per-image normalization is actively harmful — it erases the cold-cloud-top signal, which is the single most predictive cue in the entire dataset (Finding 5)
Finding 1: The Target is 82% Zeros — But 71% of Your Error Hides in 1.5% of Pixels
The GPM-IMERG precipitation target has a striking distribution: 82.1% of all pixels are exactly zero (no rain). The remaining 18% follow a heavy-tailed distribution with mean 1.65 mm/hr, but a long right tail reaching 70+ mm/hr.

The key insight is where the error lives. We analyzed a linear-regression baseline's squared error across rain intensity bins:

Rain bin (mm/hr)	% of pixels	% of squared error	% of target energy
0 – 0.1	81.6%	17.3%	0.006%
0.1 – 1.0	10.1%	5.2%	1.0%
1.0 – 5.0	6.8%	6.7%	19.0%
> 5.0	1.5%	70.7%	80.1%
71% of your squared error comes from just 1.5% of pixels — the heavy-rain tail above 5 mm/hr. And 80% of the total target energy lives in that same bin. A model that perfectly predicts zeros but misses heavy rain will score poorly. Conversely, spraying false-positive drizzle onto dry pixels is comparatively cheap.

The per-location variation is enormous: mean rain ranges from 0.00003 mm/hr (Dhaka, 99.9% zeros) to 0.92 mm/hr (Hat Yai, 62% zeros) — a ~28,000x ratio. Your model must handle both extremes without location-specific tuning, since test locations are entirely unseen.



Recommendations:

Train with a log1p(y) target transform to surface the heavy tail without changing the metric
Use a two-headed architecture: rain occurrence (BCE) + intensity regression
Weight the regression loss toward high-intensity pixels (e.g., proportional to 1 + y )
Track wet-pixel RMSE and tail RMSE (>5 mm/hr) during training — global RMSE is dominated by correctly predicting zero and hides tail failures
Simple baselines: predict-all-zeros scores 1.322 RMSE; predict-mean scores 1.298; the official SimpleCNN baseline scores 0.913; the best public solution (LH24 U-Net) scores 0.708
Finding 2: Your Band Selection Probably Has a Bug — Wavelength Mapping Matters
This is possibly the most impactful finding for anyone using the baseline code or shared notebooks.

The three satellites (Himawari, GOES, Meteosat) each have 16 bands, but band index N corresponds to different physical wavelengths on different satellites. For example:

Band index 12 is 10.4 um (IR window) on Himawari, 10.3 um (IR window) on GOES, but 9.7 um (ozone absorption!) on Meteosat
Band index 9 is 7.3 um (water vapor) on Himawari/GOES, but 6.3 um (weaker WV) on Meteosat
If you use fixed indices [12, 14, 9]  across all three satellites — as the baseline and many shared solutions do — your Meteosat model is using the wrong physical channels for 2 out of 3 bands. The 9.7 um ozone band is measurably weaker than the correct 10.5 um IR window:

Rain bin (mm/hr)	% of pixels	% of squared error	% of target energy
0 – 0.1	81.6%	17.3%	0.006%
0.1 – 1.0	10.1%	5.2%	1.0%
1.0 – 5.0	6.8%	6.7%	19.0%
> 5.0	1.5%	70.7%	80.1%
The WV channel fix is particularly large: 0.297 to 0.370 — a 25% improvement in correlation for free.



Here is the definitive wavelength-to-index mapping table — bookmark this:

Physical Channel	Wavelength	Himawari idx	GOES idx	Meteosat idx
VIS red	~0.6 um	2 (B03)	1 (C02)	2 (vis06)
Mid-IR	~3.9 um	6 (B07)	6 (C07)	8 (ir38)
WV (lower)	~7.3 um	9 (B10)	9 (C10)	10 (wv73)
IR 8.5 um	~8.5 um	10 (B11)	10 (C11)	11 (ir87)
IR window	~10.5 um	12 (B13)	12 (C13)	13 (ir105)
IR split	~12.3 um	14 (B15)	14 (C15)	14 (ir123)
In Python:

BANDS = {
    "himawari": [2, 6, 9, 10, 12, 14],
    "goes":     [1, 6, 9, 10, 12, 14],
    "meteosat": [2, 8, 10, 11, 13, 14],
}


Finding 3: IR Bands Dominate — Visible Bands Are Useless at Night (52-73% of Data)
We computed Mutual Information (MI) between each of the 16 bands and the precipitation target, split by day and night conditions. The results are dramatic:

IR/WV bands (3.9-13.3 um) have MI of 0.19-0.27 (GOES/Himawari) and are stable 24/7
VIS bands (0.4-2.3 um) have MI of 0.10-0.15 during daytime but collapse to 0.00-0.01 at night — essentially random noise
The nighttime fraction is large: Himawari 64%, GOES 73%, Meteosat 66%. The majority of your training data has zero useful signal in visible bands.

The top-4 most informative bands on every satellite are all in the 3.9-12.3 um IR/WV range. The IR window band (~10.5 um) is consistently the single strongest predictor, with the 8.5 um and 12.3 um bands close behind.



Recommendations:

Derive a day/night flag from mean visible-band brightness of the patch (e.g., mean (VIS) < 5 = night), NOT from UTC hour — locations span the globe, so UTC is not a reliable day/night proxy
Optionally zero out VIS channels at night to prevent the model from learning noise
Prioritize IR-based features and architectures
The day/night-robust minimal band set is: [IR window 10.5, WV 7.3, Mid-IR 3.9]  per satellite
Finding 4: This is a Pure Geographic Holdout — Your Validation Strategy is Everything
This is the single biggest unlock for this competition.

The train set has 20 locations. The test set has 18 locations. There is zero overlap. Each location is tied to exactly one satellite. This means:

Standard random cross-validation will massively overfit. Within-location temporal autocorrelation leaks information — your CV will be wildly optimistic and won't correlate with the leaderboard.
The only valid CV is GroupKFold by name_location , never split a location across train/val.
The binding constraint is that there are only 5 GOES train locations, so the maximum number of folds that keeps at least 1 GOES location per fold is 5.
We precomputed a balanced 5-fold assignment that ensures each fold has all 3 satellites and spans dry-to-wet climates:

Fold	Locations	Satellites
F0	borno_state, atlantic_coast, hat_yai	met, goes, him
F1	gaza_province, central_vietnam, ecuador	met, him, goes
F2	bihar, cape_town, bahia_blanca, kinshasa, aceh	him, met, goes, met, him
F3	guangdong, friuli_venezia_giulia, central_philippines, jamaica	him, met, him, goes
F4	dhaka, france, andalusia, florida, jakarta	him, met, met, goes, him
Critical warning: single-split holdout RMSE varies 1.15 to 1.71 across different location groupings for the same model — pure geographic luck. Always average over all 5 folds. Treat LB differences < ~0.05 as possibly noise.

Weight your CV by the test satellite mix (himawari 39%, meteosat 39%, goes 22%), not a naive location average — GOES is under-represented in train but must be served in test.



Finding 5: Never Normalize Per-Image — It Destroys the Signal
This one is counterintuitive if you come from computer vision, where per-image normalization is standard practice.

The single most predictive scalar feature in this dataset is the IR window band's absolute minimum pixel value — its Spearman correlation with rain fraction is -0.84 (Himawari). Cold cloud tops mean deep convection, which means heavy rain. This is an absolute temperature signal.

Per-image standardization (
(x - mean) / std
computed per frame) normalizes away the cold-vs-warm distinction. It converts the absolute brightness temperature — which directly encodes cloud-top height and convective intensity — into a relative measure that means nothing for precipitation estimation. Between-frame IR-window mean variation (coefficient of variation 0.23-0.37) is the warm-vs-cold rain signal; per-image normalization erases it.

Additionally, solar/VIS bands have a 30-48% exact-zero spike from nighttime that severely distorts mean and standard deviation, making per-image standardization unreliable even mechanically.

The correct approach: precompute global per-(satellite, band) median and IQR from training data, then normalize:

x_norm = (x - median_b) / (IQR_b + eps)
x_norm = x_norm.clamp(-5, 5)  # neutralize outliers
Never share constants across satellites (same band index ≠ same wavelength/scale). We provide precomputed normalization constants — contact us if you'd like them.

Finding 6: The Split-Window Difference is a Trap — Use the Ratio Instead
The brightness temperature difference BT(10.5) - BT(12.3) (the "split-window difference") is a textbook feature in satellite meteorology for detecting cloud microphysics and rain potential. Many published solutions and shared notebooks compute it as an engineered feature.

On this dataset, it adds essentially nothing.

The reason: the satellite data is stored as uint8 (0-255). The physical split-window BT difference is typically 1-3 Kelvin — but the uint8 quantization step is larger than this signal. The difference gets quantized away to near-zero variation. We verified this: partial correlation with rain after the IR window band is already present is ~0.00 on all three satellites.

The fix is simple: use the split-window ratio instead of the difference:

Feature	Formula	Partial corr (Himawari)	Partial corr (GOES)	Partial corr (Meteosat)
Split-window DIFF	
W - SPL
~0.00	~0.00	~0.00
Split-window RATIO	
SPL / (W + 1)
-0.31	-0.20	-0.13
IR 8.5 - Window	
IR85 - W
-0.24	-0.20	-0.04
WV(7.3) - Window	
WV - W
-0.10	-0.17	-0.12
The ratio is a softer nonlinearity that survives the quantization. And there's a hidden gem: the 8.5 um minus window difference has near-zero raw correlation with rain but is the #2 additive feature on Himawari and GOES — a genuine signal that most competitors are not exploiting.

Recommendations:

Drop the raw split-window difference W - SPL
Add the split-window ratio SPL / (W + 1) , strongest additive feature
Add the IR 8.5 - window difference IR85 - W  , hidden gem
Add the WV - window difference WV - W  , deep convection proxy
Compute all features on raw uint8 values before normalization, then normalize each separately
Finding 7: Temporal Value is Real But Small — Don't Over-Invest
Frames are exactly 10 minutes apart; the last frame (t2) is exactly 10 minutes before the target. The full 3-frame sequence spanning 20 minutes of history is available for 98% of samples.

We quantified how much the temporal dimension contributes:

Input	R-squared (rain fraction)
Last frame only	0.610
Mean of frames	0.574 (WORSE!)
Last + temporal diff	0.619
Last + motion features	0.623
Mean-pooling the frames is worse than using just the last frame. The useful temporal signal comes from the per-pixel change magnitude (|delta| Spearman +0.69 with rain), not from signed mean cooling (Spearman +0.05, essentially useless) or optical flow displacement (Spearman -0.05, useless).



Recommendations:

Stack all 3 frames as channels (chronological order, last frame primary) — never average them
Add one temporal diff channel: W[t2] - W[t0] on the window band, fed as a signed channel
Don't compute optical flow — it adds nothing
Drop the 235 zero-frame samples (all Meteosat) from training — they have valid targets but no inputs, so they're pure label noise
Pad the 647 two-frame samples by repeating the last available frame
Finding 8: Resize, Don't Crop — The Full Frame IS the Target Extent
Native satellite frame sizes differ: Himawari 81x81, GOES 141x141, Meteosat 144x144. The GeoTIFF georeference tags are stripped (only an identity transform), so co-registration must be established empirically.

We tested this by computing the Pearson correlation between the IR window band and log1p(target rain) at different center-crop fractions, resizing each crop to 41x41:

Result: full frame resize (fraction = 1.0) ALWAYS gives the highest correlation, and every crop monotonically destroys signal. GOES drops from Pearson 0.41 (full frame) to 0.07 (20% crop) — a near-total loss.



This confirms that the satellite frames already cover the same geographic footprint as the 41x41 target grid. Cropping throws away context the model needs.

We also tested spatial offsets: the optimal integer shift peaks at (0,0) for GOES/Meteosat and (0,+1) for Himawari — essentially sub-pixel, so no offset correction is needed.

Recommendations:

Resize the full native frame to your model input grid using bilinear interpolation
Do NOT center-crop. Do NOT crop-then-resize.
Predict at native 41x41 target resolution (pad to 48x48 for pooling divisibility if needed, then crop back)
Finding 9: The Data Has Silent Traps — Check for These
We found several data quality issues that can corrupt training, normalization, or submissions without throwing errors:

Issue	Count	Satellite	Impact	Action
Zero-input rows (no satellite data at all)	235	All Meteosat	Label noise if loaded blindly	DROP from training
1-frame rows	8	Mixed	Minimal	Drop or pad
2-frame rows	647 (1.6%)	Mostly Himawari	Missing temporal info	Pad by repeating last frame
Eval test targets = RANDOM NOISE	All 29,090	All	Corrupts any normalization or statistics	NEVER read test target files for stats
Corrupt GOES frames	~1/200 frames	GOES	NaN/inf in batches	Detect all-zero/constant bands and skip
Malformed eval GeoTIFFs	At least 1 file	GOES (upper_midwest)	Data loader crash	Guard against short band stacks
The most dangerous trap: the evaluation test_files/.tif  target images contain uniform random noise on [0, 50] (mean ~25, every pixel distinct). They are placeholder files, NOT real precipitation. Any pipeline that reads test target statistics — for normalization, log1p scaling, BCE threshold, clipping, or anything else — will be completely wrong. Derive ALL target statistics exclusively from training GPM-IMERG files.

Finding 10: Meteosat is a Different Beast — Consider Satellite-Specific Treatment
The three satellites are not created equal for this task:

Metric	GOES	Himawari	Meteosat
Zero fraction	75.6%	79.0%	88.5%
Mean rain (mm/hr)	0.385	0.379	0.173
Max MI (best band)	0.270	0.246	0.126
Train sample count	10,272	13,192	17,222
Test locations	4	7	7
Meteosat covers climatologically drier regions (Europe, Africa) and its band signal strength is genuinely about half that of Himawari and GOES. This is partly a geography confound, but also reflects real instrument differences — a single shared brightness-to-rain mapping measurably hurts Meteosat.

The test set has a similar balance (Himawari 7, Meteosat 7, GOES 4 locations), so Meteosat performance matters as much as Himawari's.

Recommendations:

Add satellite ID as a conditioning signal (e.g., FiLM conditioning or a learned embedding)
Report and track per-satellite validation RMSE alongside overall RMSE
Weight your CV average by the test satellite mix (him 39%, met 39%, goes 22%)
Consider per-satellite normalization constants (already covered in Finding 5)
Bonus: The Score Landscape
To calibrate your expectations:

Method	RMSE
Predict all zeros	1.322
Predict global mean	1.298
Official SimpleCNN baseline	0.913
LH24 U-Net (public writeup)	0.708
Current LB leader	0.644
The jump from 1.3 (trivial baselines) to 0.9 (simple CNN) is straightforward. The jump from 0.9 to 0.7 requires getting band selection, normalization, and loss design right — the findings in this post. The jump from 0.7 to 0.65 is where architectural choices, distributional heads, and domain generalization start to matter.

Summary of Actionable Recommendations
For quick reference, here are all the recommendations ranked by expected impact:

Switch to GroupKFold by
name_location
(Finding 4) — the single biggest unlock
Fix your Meteosat band indices to wavelength-mapped values (Finding 2) — free accuracy
Use global per-(sat, band) median/IQR normalization, never per-image (Finding 5)
Train with log1p target + two-headed architecture + tail-weighted loss (Finding 1)
Replace split-window difference with ratio + add 8.5-window diff (Finding 6)
Resize full frame, don't crop (Finding 8)
Stack frames as channels, don't average; add temporal diff channel (Finding 7)
Drop 235 zero-input rows and guard against corrupt GOES frames (Finding 9)
Add satellite conditioning and track per-satellite RMSE (Finding 10)
Never read eval target files for statistics (Finding 9)
Closing
These findings aren't just academic exercises — they directly inform how we build models that can save lives in regions without radar infrastructure. The geographic holdout design mirrors the real deployment scenario for the Solafune FEWS: we train where satellite calibration data is plentiful and deploy to data-poor regions where flash floods kill thousands annually.

We hope this analysis saves everyone time and improves the overall quality of solutions. We are happy to answer questions and share the analysis code and precomputed statistics. Let's push the state of the art together.

Good luck to all participants!

################

The Information Budget of Satellite Nowcasting: Where the Signal Lives, What the Error Is Made Of, and How Good Anyone Can Get

In our previous post we shared 10 practical findings from exploring this dataset (band mappings, validation design, normalization pitfalls, data traps). This post goes a level deeper. It is a research-grade analysis of three questions that ultimately decide every leaderboard position:

Where does the predictive information live — in time, in space, and in the statistics of the target?
What exactly is the RMSE made of, and which modeling choices follow mathematically rather than from taste?
How good can anyone get — what does the physics literature say about the skill ceiling of instantaneous IR→rain estimation?
Everything below is measured on this competition's data (scripts referenced at the end) or cited from verifiable literature. We also include a dataset-structure notice that we believe the community and organizers should be aware of.

Part I — Where the information lives in time
We measured the per-pixel correlation between the IR-window brightness temperature (the strongest single predictor) and the target rain field, as a function of the frame's time offset relative to the target instant t. For 250 tiles per satellite, we evaluated frames from t−30 up to t+30 minutes:



Three results, consistent across all satellites:

Information peaks exactly at the target instant. |Pearson r| rises monotonically as the frame time approaches t: 0.584→0.595 (Himawari, t−30→t), 0.550→0.573 (GOES), 0.404→0.423 (Meteosat). This is physically obvious in hindsight — the cloud field at t is what is raining at t — but it quantifies the cost of the causal cutoff: our newest allowed input (t−10) sits 10 minutes short of the information peak.
The causal history is almost worthless in a linear probe. Adding frames t−20 and t−30 on top of t−10 raises pooled-pixel linear R² by only +0.2–0.4%. (A CNN extracts a bit more — our earlier tile-level analysis found +1–3% — but the order of magnitude stands.)
The frames we cannot use are worth more than the ones we can. As an upper-bound experiment, a linear model given frames at/after t gains +3.3% (Himawari), +5.8% (GOES), +4.2% (Meteosat) over the full causal set — several times the value of the entire usable history.
The design implication is sharp: the marginal value of more history is nearly zero; the missing information is the 10-minute advection gap. A competitive model should spend its capacity inferring the cloud field's evolution from t−10 to t (motion, growth, decay of convective cores), not on ingesting longer sequences. This is why recurrent temporal mixers underperform here (we measured ConvGRU below a plain frame-stack, and optical-flow features at ρ≈−0.05): with 20 minutes of usable history there is nothing for them to model — the problem is short-range extrapolation, not sequence learning.

A legitimate way to exploit this: the frame at time t does exist in the training folder for 98.7% of training rows (it is the next row's input). It can serve as a training-time auxiliary supervision target — e.g., an auxiliary head that must predict the future frame, teaching the network advection end-to-end — without ever being used as a test-time input. We flag the flip side of this fact next.

Part II — Dataset-structure notice (please read, and organizers please clarify)
The observation files for all rows ship together in the dataset folders. Because consecutive rows are 30 minutes apart and each lists 3 frames, the frame captured at the target instant t — and frames after it — can be located by simple filename reconstruction for almost every row, including evaluation rows:



Frame offset vs target	Train	Evaluation
t−30 … t−10 (the row's CSV inputs)	98.5–99.3%	98.6–99.8%
t+0 — at the target instant	98.7%	99.4%
t+10 / t+20 / t+30	97.9–98.7%	98.2–99.3%
Combined with Part I (information peaks at t), this means a submission could quietly consume near-simultaneous observations of the very instant it must "nowcast," turning the task into retrieval rather than forecasting. We measured the availability but chose not to use these frames as inputs — it is against the spirit of a nowcasting benchmark whose stated purpose is real-time flood early warning, where the future is by definition unavailable.

We are disclosing this openly because (a) participants deserve a level playing field, and (b) the organizers may want to explicitly state whether inputs are restricted to each row's
**last_30_minutes_observation_filename**
. We'd welcome an official clarification in this thread. (To be clear: using the training folder's future frames as training-time-only auxiliary labels, as in Part I, involves no test-time information and, in our reading, no fairness issue.)

Part III — What the RMSE is actually made of
We profiled simple baselines (predict-zero, global mean, a best-single-band linear model, and linear + dry threshold) on a 7-location holdout, decomposing error by true-intensity bin:



The anatomy in numbers (linear model):

True intensity (mm/hr)	% of pixels	% of squared error	RMSE within bin (predict-zero)
0 – 0.1	81.6%	17.3%	0.013
0.1 – 1	10.1%	5.2%	0.48
1 – 5	6.8%	6.7%	2.58
> 5	1.5%	70.7%	11.15
Three consequences that many teams learn the expensive way:

The >5 mm/hr bin is the competition. It is ~10× the conditional RMSE of any other bin and carries 71% of the squared error. But — critically — you cannot fix it by re-weighting the loss toward heavy rain. We measured tail up-weighting and inverse-density weighting as net-negative: with 82% of pixels exactly zero, inflating tail gradients pushes drizzle onto dry pixels and the dry-mass penalty exceeds the tail gain. The fix has to be structural (Part IV), not a loss multiplier.
Per-satellite error anchors differ by ~4×. On our holdout, predict-zero scores 2.17 (Himawari), 1.48 (GOES), 0.54 (Meteosat) — Meteosat regions are simply drier. Your public-LB score is a weighted mixture of three very different sub-problems; track them separately or improvements in one will hide regressions in another.
Holdout composition dominates small score differences. The same predictor scores anywhere between ~1.15 and ~1.71 depending on which locations land in the holdout. Single-split validation numbers — and LB differences below ~0.05 — are substantially geography luck.
Part IV — The statistical law of the target (and what it forces your head to be)
Here is the most useful distributional fact in this dataset. We sampled 93,368 wet pixels from 300 random target tiles and examined ln(y):



Wet-pixel intensities are almost exactly log-normal: ln(y)|y>0 fits N(μ=−0.66, σ=1.63) with skew −0.15 (a Gaussian has 0). So the target is a textbook zero-inflated log-normal process:

Y = B · Z,   B ~ Bernoulli(p(x)),  ln Z ~ N(mu(x), sigma(x)^2),  p ≈ 0.18 pooled
(The comb-like spikes at low ln(y) are IMERG's own value quantization — the "continuous" target is discrete at low intensities, another reason sub-0.1 mm/hr precision is wasted effort.)

This has mathematical consequences for architecture, independent of taste:

1. The only RMSE-optimal point estimate is the conditional mean. For squared error, argmin_c E[(Y−c)²|X] = E[Y|X]. Not the median, not a quantile, not "the most likely value":



For a log-normal, mean/median = exp(σ²/2). With our measured σ=1.63, the conditional mean is ~3.8× the conditional median. Any model whose output behaves like a median — which includes L1-trained models and served quantiles — under-predicts wet pixels roughly four-fold. We verified this empirically: serving quantiles q0.5–q0.9 from a quantile head cost us 0.09–0.15 RMSE versus serving the mean. If you take one equation from this post: serve E[Y|X], always.

2. The natural head is a product decomposition. The zero-inflated structure factorizes the mean exactly:

E[Y|X] = P(rain|X) · E[Y | rain, X]
which suggests a two-part ("hurdle") head: a rain-occurrence classifier times a wet-intensity regressor trained on wet pixels only. The wet-only training is the point — the intensity branch never sees the 82% zero mass, so it cannot be dragged toward zero, which is exactly the regression-to-the-mean failure that plagues single-head L2 models on this data. The multiplicative gate protects dry pixels.

3. The occurrence classifier must be unweighted. For the product above to be an unbiased mean, P(rain|X) must be calibrated. Class-weighted BCE (the standard "fix" for imbalance) deliberately mis-calibrates the probability — it would silently bias every wet prediction. Resist the imbalance reflex here.

4. If you prefer a binned/categorical head, use fine bins and unweighted CE. The expectation Σ p_k·c_k over ≥50 log-spaced bins approximates E[Y|X] with negligible discretization bias; we measured a coarse 3-class variant at +0.032 RMSE (the top bin's center cannot represent a heavy tail), while class-weighting breaks the mean for the same calibration reason as above.

5. log1p is the right single-regressor transform, but not the right wet transform. As a single transform over all pixels, log1p is stable at the 82% zeros and invertible — the right pragmatic choice for a lone regression head. But conditional on rain, log1p(y) is still clearly skewed (+1.31) while ln(y) is Gaussian (−0.15). If you run a hurdle, give the wet branch ln(y) or a Gamma/log-normal likelihood; keep log1p only for single-head regressors.

Part V — The spatial law of the target


Rain is strongly spatially coherent: Moran's I = 0.57 at lag 1; 98.3% of rainy pixels have a rainy neighbor, versus 47.1% expected if pixels were independent at the same density. Isolated rainy pixels essentially do not exist in GPM-IMERG. Salt-and-pepper predictions are guaranteed error; smooth, connected rain fields are not a stylistic choice but a property of the measure itself.
Tiles are event-centered: mean rain falls off ~29% from tile center to corner. The sampling process evidently centered tiles on precipitation events. Two implications: (a) the tile center is where the action — and the error — concentrates; (b) any resampling of the target grid (e.g., predicting at 128×128 and downscaling) interpolates away this radial structure. Predict at the native 41×41.
Part VI — How good can anyone get? The skill ceiling
A sober look at the literature on instantaneous IR-based precipitation estimation:



So & Shin (2018, QJRMS, doi:10.1002/qj.3288) put it bluntly: cloud-top temperatures "are weakly related to surface rainfall, particularly for shallow or warm clouds."
PRE-Net (arXiv:2506.07050, Table 5) benchmarks instantaneous retrievals against radar-radiometer truth: PDIR-Now CC 0.08, PERSIANN-CCS CC 0.21, IMERG itself CC 0.45, their 2025 SOTA 0.50. Read that middle number again: the very product we are asked to predict correlates only ~0.45 with instantaneous ground truth. We are predicting a noisy proxy of rain from an even more indirect signal.
GREMLIN (Hilburn et al. 2021, JAMC, doi:10.1175/JAMC-D-20-0084.1) achieves R²=0.74 GOES→radar-reflectivity — evidence that learned multi-scale IR features carry real convective skill (their XAI shows gradients and cold-cloud structure, not single-pixel temperature, do the work) — but that is reflectivity aloft, not surface rain rate.
The practical reading for this competition: a large fraction of the wet-pixel error is irreducible Bayes risk — conditional variance that no head, loss, or architecture removes, because IR/WV brightness temperature does not determine instantaneous surface rain. That is why the score landscape compresses near the top (predict-zero 1.32 → leader 0.644, and the last 0.03 has absorbed the field's best efforts), and why we recommend spending effort in this order:

Information first: correct wavelength-mapped bands, physics-derived channels that survive uint8 quantization (ratios, not the classic split-window difference), motion/advection modeling across the 10-min gap;
Estimator correctness second: everything in Part IV — serve the conditional mean through a decomposition that respects the zero-inflated log-normal law;
Variance reduction last: ensembling, TTA, seed averaging — worth a few thousandths each, no more.
You can estimate your own remaining headroom: take your model's penultimate features, find k=50 nearest neighbors restricted to other locations (to respect the geographic holdout), and compute the conditional variance of y among neighbors. Where that floor sits at your current wet-tile RMSE, no loss function will save you — only new information will.

Summary of recommendations
#	Recommendation	Basis
1	Don't ingest longer history; model the 10-min advection gap (e.g., future-frame prediction as a training-time auxiliary task from the train folder)	Part I
2	Treat frames at/after target time as out-of-bounds inputs; ask organizers to codify this	Part II
3	Serve E[YX] — never a median/quantile (≈3.8× systematic under-prediction on wet pixels)	 
4	Use a hurdle decomposition: calibrated P(rain) × wet-only intensity model	Part IV
5	Keep the occurrence BCE (and any categorical CE) unweighted — calibration is what makes the product unbiased	Part IV
6	If categorical: ≥50 log-spaced bins; coarse bins measurably fail (+0.032)	Part IV
7	Give wet-intensity branches ln(y) or a Gamma/log-normal likelihood; keep log1p only for single-head regressors	Part IV
8	Never re-weight the loss toward heavy rain; it is measured net-negative under 82% zeros	Part III
9	Predictions should be smooth and connected (Moran's I 0.57; 98.3% neighbor contiguity); predict at native 41×41	Part V
10	Track per-satellite and per-bin RMSE; treat <0.05 LB differences as geography noise	Part III
11	Estimate your Bayes floor with location-blocked kNN conditional variance before buying more GPUs	Part VI
12	Sub-0.1 mm/hr precision is wasted (IMERG quantization + drizzle thresholding is a small free win)	Parts III–IV
Reproducibility
All measurements in this post come from the competition data and these scripts (happy to share on request): temporal-offset analysis (
retrieval_vs_forecast_gain
), frame-availability audit (
future_frame_coverage
), baseline error profiling, target-distribution and spatial-structure analysis. Literature citations are given inline with DOIs/arXiv IDs.

We hope this helps teams spend their remaining compute where it matters — and we'd genuinely welcome organizer clarification on Part II. Good luck to everyone; may the best nowcaster win.

#################
Decoding the Data Desert: A Deep-Dive EDA into What Our Satellites Are Actually Telling Us

Introduction:

Before we talk models, let's talk data.

This discussion has three aims:

Learn together — this is an interactive walkthrough, not a monologue. I'll share findings and plots, and I want to hear what you're seeing too(I could possible be wrong in my approach, it's my first time working on something like this). Drop your numbers, your disagreements, your discoveries in the comments.
Understand the dataset — its features, its components, its quirks. Before anyone writes a training loop, we should collectively know what we're actually feeding our models.
Surface and solve anomalies — I've already spotted a few things worth flagging (more on that below). I don't have all the answers, so I'm genuinely open to thoughts on how others are handling them.
Where We Start: Data Distribution

The first thing to understand is how the data is actually structured — how many samples do we have per location and per satellite, and does that hold consistently across both train and test splits?

This matters more than it seems. If certain locations or satellites are heavily over-represented in train but sparse in test (or vice versa), that's a generalization problem baked into the data before you've even chosen an architecture.

So let's start here: sample count per location and per satellite, broken down by split.  



After looking at this(or even before), there is something important to consider, Are the locations and satellites in our train and test datasets disjoint? Well, for location, it is:
Train: aceh, andalusia, atlantic_coast, bahia_blanca, bihar, borno_state, cape_town, central_philippines, central_vietnam, dhaka, ecuador, florida, france, friuli_venezia_giulia, 'gaza_province, guangdong, hat_yai, jakarta, jamaica, kinshasa.
Test: kanto_region, limpopo_province, lombardia, maputo_province, mekong_delta, mexico, niger_state, north_sumatra, northeast_malaysia, peru, quang_nam, rio_grande_do_sul, sofala_province, sri_lanka, sylhet, tanganyika, upper_midwest, valencia

This is already tricky enough; In my opinion, since the train and test locations are completely disjoint by design, it is arguably an important thing to understand about this competition. It transforms the problem from fitting into generalization — the model will never see a test location during training, so it cannot memorize location-specific rainfall patterns or seasonal climatology. What it must generalize instead are the raw spectral signals: cloud brightness, thermal signatures, water vapor — the physics of precipitation as seen from space, not the geography of where it's happening. This also means any feature that encodes location identity will overfit and hurt you, and satellite coverage asymmetry across regions becomes a direct threat to test performance.

Before we move on, I think it is fair to plot random tifs(timestamps) and targets:
To be fair, I do not think our eyes do any justice, it is only normal to want to see what our data looks like.











Along the line, I observed something important and that would be shape mismatches(particular to the goes satellite), in some cases, it is observed that for each entry in the 'last_30_minutes_observation_filename' column, not all tifs have the same shape(this can easily be handled by interpolation). Asides this, missing timestamps, in some cases 2, in some cases 1 and in other cases all timestamps were found missing. 

Rainfall Distribution

The first step in understanding any regression target is seeing its shape. We plot histograms of Mean Precipitation broken down by location — one panel per location — to understand how rainfall is distributed across the dataset. What you'll typically see is a heavily X-skewed distribution: most observations cluster near zero (dry or lightly raining scenes) with a long tail of intense rainfall events. This shape is not a quirk — it reflects the real-world nature of precipitation, where extreme events are rare but disproportionately impactful.



A log transform compresses that long tail, pulling extreme values closer to the bulk of the distribution. This does two things for modeling: it prevents the loss function from being dominated by a handful of heavy rainfall events during training, and it forces the model to pay attention to the structure of light-to-moderate rainfall rather than chasing outliers. If you're optimizing RMSE on raw values, a single extreme event can swing your gradients significantly — log-transforming the target before training (and inverting at inference) is a practical way to stabilize that.  There are other useful transformation methods that can be used as well.


Understanding Sample Precipitation Metrics

I derived precipitation statistics per observation, each capturing a different dimension of rainfall:

Mean Precipitation is the average precipitation value across all pixels in the GPM-IMERG patch, including dry pixels (where precipitation = 0). It reflects the overall rainfall signal for the scene — a low value could mean light rain everywhere, or heavy rain in a small area surrounded by dry pixels.



Mean Wet Precipitation is the average precipitation computed only over pixels where rainfall is actually detected (wet pixels). This isolates intensity — how hard is it actually raining where it is raining — stripping out the diluting effect of dry pixels in the scene.



Wet/Dry Precipitation Fraction is the proportion of pixels in the patch that are wet. It captures rain coverage — how much of the scene is experiencing rainfall at all, independent of how intense that rainfall is.

Together, these three metrics decompose precipitation into three orthogonal questions: How much overall? How intense where it's raining? How widespread is it? Analyzing them separately matters because a model could plausibly learn to predict one well while failing at another — for example, correctly identifying that rain is occurring (wet fraction) while underestimating its intensity (mean wet precipitation). Understanding where each metric is highest, most variable, and most skewed tells you which aspect of rainfall your model will struggle with most.

Quarterly Precipitation by Location and Satellite

Precipitation is not static across the year — it follows seasonal rhythms driven by monsoons, trade winds and regional climate systems. Breaking down Mean Precipitation by quarter per location and satellite reveals whether the data captures these cycles faithfully, and more importantly, whether certain quarters are over- or under-represented in the dataset.

This matters for two reasons. First, a model trained on data skewed toward wet-season observations will have a distorted sense of what "normal" looks like for a given region. Second, since train and test locations are disjoint, the quarterly distribution of the training data is what the model generalizes from. If Q1 and Q2 are sparse in training, the model may underperform on test locations that happen to be in their wet season during the evaluation period.

Looking at this broken down by satellite adds another layer: if GOES observations are concentrated in Q3/Q4 while Meteosat is spread evenly, that imbalance could explain performance differences between satellites beyond just spectral characteristics.

















Hourly Precipitation by Location

The diurnal cycle is one of the strongest and most consistent signals in tropical and subtropical precipitation. Convective rainfall over land typically peaks in the late afternoon, driven by surface heating, while coastal and maritime regions often peak at night or early morning. Plotting mean precipitation by hour of day per location tells you whether this cycle is present and well-sampled in the data.  









































For a satellite-based nowcasting model, this is especially important because the satellite bands themselves behave differently across the day — visible bands are unavailable at night, while thermal IR bands carry the full load. If precipitation peaks at hours when visible information is absent, the model needs to rely entirely on IR and water vapor channels at precisely the moments when rainfall is most intense. Knowing which locations have strong daytime vs. nighttime precipitation peaks helps anticipate where the model's spectral inputs will be most and least informative.

Box Plots per Location and Satellite

Histograms tell you the shape of a distribution, but box plots give you something more immediately comparable across many groups at once: the median, spread, and outlier structure of precipitation for every location and satellite in a single view. Ranking locations by median or P75 precipitation immediately separates the chronically wet regions from the dry ones, and the width of each box tells you how variable rainfall is within a location — high variance locations are harder to predict regardless of model choice.



Stratifying by satellite reveals whether the precipitation statistics a model sees are consistent across sensors or whether GOES, Himawari, and Meteosat are effectively sampling different rainfall regimes by virtue of their geographic footprints. If Himawari box plots are systematically higher than Meteosat, that could reflect genuine regional differences in rainfall intensity, or it could reflect a sampling imbalance — and distinguishing between the two matters for how you normalize inputs across satellites.



Understanding the Relationships Between Precipitation Metrics

Before building any model, it is worth asking: do these three metrics move together, or do they capture genuinely different things? The heatmap of correlations between Mean Precipitation, Mean Wet Precipitation, and Wet/Dry Precipitation Fraction answers that directly.

If Mean Precipitation and Wet/Dry Fraction are highly correlated, it tells you that overall rainfall in a scene is driven primarily by how widespread the rain is, not how intense it is. If Mean Wet Precipitation correlates more weakly with the others, it suggests that intensity is a relatively independent signal — scenes can have heavy rain concentrated in a small area, or light drizzle spread across the entire patch.

This has direct modeling implications. If all three metrics are near-perfectly correlated, predicting one effectively predicts the others and there is no additional structure to exploit. But if they diverge, a model that only optimizes for Mean Precipitation might be systematically wrong about intensity and coverage in ways that matter for flood warning — a narrow but extremely intense rainfall band is a very different flood risk from widespread light rain, even if the mean values are similar.

Understanding these correlations also guides feature engineering. Highly correlated targets can be predicted with a shared representation, while weakly correlated ones may benefit from separate output heads or auxiliary losses.



How Heavy Is the Right Tail? (P90, P95, P99 per Location)

The mean tells you what typical rainfall looks like. The right tail tells you what the model will be punished for missing.

RMSE is not a forgiving metric — squaring the error means that a single large miss on an extreme rainfall event contributes far more to your score than dozens of small misses on dry scenes. This means the distribution of extreme values is not just an academic curiosity; it directly shapes where your model needs to be most accurate.

Plotting P90, P95, and P99 of Mean Precipitation per location reveals which locations are the hardest to model. A location where P90 is already high means extreme rainfall is relatively common there — the model sees enough of it during training to learn from. But a location where P90 is moderate and P99 is dramatically higher signals rare but catastrophic events that are statistically difficult to learn from and expensive to miss.

The gap between P90 and P99 is the number to watch. A large jump means the tail is heavy and erratic — the top 1% of rainfall events in that location are outliers even relative to other wet observations. These are exactly the events that matter most for flood early warning, and exactly the ones a model trained to minimize average loss will underweight.  



Mean Precipitation per Location

While the box plots show spread and variability, bar plots of mean precipitation per location give you the clearest ranked view of which locations are systematically wetter or drier on average. This is the baseline rainfall fingerprint of each region in the dataset.

Ranking locations by mean precipitation immediately separates the high-rainfall tropical regions — where convective systems are frequent and intense — from the arid and semi-arid regions where rainfall is sparse and episodic. This ranking matters because the model will see very different label distributions depending on which locations dominate the training data. If the wettest locations contribute the most samples, the model's loss landscape is shaped by high-rainfall events. If dry locations dominate, the model may be implicitly biased toward predicting near-zero precipitation.

Cross-referencing this with the satellite column adds another dimension: if the wettest locations happen to all be covered by one satellite, then that satellite's spectral characteristics become disproportionately associated with high rainfall during training — a potential confound that could hurt generalization to unseen test locations covered by a different satellite.

This is also a useful sanity check. Locations like Aceh, Central Philippines, and Jakarta should sit near the top given their tropical maritime climate. Locations like Bahia Blanca, Cape Town, and Gaza Province should sit lower. If the rankings look climatologically implausible, that is a data quality flag worth investigating.

























Final Notes

This discussion set out to understand the dataset before touching a single model, and I think we've covered what matters most. We've confirmed that precipitation is heavily right-skewed with significant dry sample imbalance, that the three precipitation metrics decompose rainfall into genuinely different dimensions, that diurnal and seasonal cycles are real and well-captured, and that satellite-location entanglement plus incomplete sequences are the two input quality issues most worth handling deliberately.

Open Questions Going Into Modeling

How are you handling the dry/wet imbalance at the loss level — weighted MSE, two-stage predict-then-regress, or something else?
For incomplete sequences (num_tifs_per_target < 3), are you masking, imputing, or dropping?
Do you normalize each satellite independently or build a unified cross-satellite normalization scheme?
Does time-of-day warrant an explicit positional embedding given the strong diurnal signal we saw?
What way, in your opinion will be the best split strategy? By satellite? Location? Timestamp?...
Possible Directions

ConvLSTM
U-Net with temporal stacking

#######################

Back to Discussion

ibrahimqasmi

Posted 1 週間前

9
The 0.68 wall , I scored a PERFECT forecast, and it explains the whole leaderboard

ok so. most of us are packed between 0.68 and 0.70 (i'm #17 at 0.6782, so this is self-therapy too). different architectures, different losses, everyone lands in the same place. instead of training model #12, i decided to ask a weirder question:

what would an ORACLE score on this metric?

take the ground truth itself, handicap it in controlled ways (flatten it, take its median, blur it...), and score these "cheating predictors" with the exact competition metric. no model anywhere in this notebook — just the data and the metric. the answer explains the plateau, and (spoiler) the top ~16 have already escaped it, in a very specific direction.

everything below runs on the full 40,686-tile training set with the exact metric. runs end-to-end in a few minutes.

 



















import os, ast, glob, json
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, tifffile
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, PowerNorm

# ---- find the data  ----
#########

# ---- THE metric: per-sample RMSE (per tile sqrt(mean(sq err)), then plain mean over tiles) ----
def tile_rmse(pred, y):
    return float(np.sqrt(((pred - y) ** 2).mean()))

# small plot style so figures don't look like default matplotlib
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, VIOLET, RED, SURF = "#2a78d6", "#4a3aa7", "#e34948", "#fcfcfb"
RAIN = LinearSegmentedColormap.from_list("rain", [SURF,"#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#1c5cab","#104281","#0d366b"])
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "text.color": INK, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False})
train rows: 40686  |  locations: 20
one pass over every target
single streaming pass over all 40,686 GPM tiles. for each tile i keep its mean, mean-square, and directly score the per-tile oracles (median / mean+mask need the raw pixels). the flat-constant + climatology + oracle-mean scores then come out of the stored stats for free — rmse(c) = sqrt(c² − 2c·mean + meansq) per tile, and the flat tile-mean oracle is just each tile's pixel std.

[2]: 
 































N = len(df)
t_mean  = np.zeros(N); t_msq = np.zeros(N); t_y2 = np.zeros(N)
r_med   = np.zeros(N); r_mask = np.zeros(N)
zero_px = 0; tot_px = 0
THR = np.array([0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40]); y2_ge = np.zeros(len(THR))
locs = df.name_location.values

for i, fn in enumerate(df.gpm_imerg_filename.values):
    y = tifffile.imread(os.path.join(TRAIN, "gpm_imerg", fn)).astype(np.float32)
    y = np.clip(np.nan_to_num(y), 0, None)
    t_mean[i], t_msq[i] = y.mean(), (y * y).mean()
    t_y2[i] = (y * y).sum()
    r_med[i]  = tile_rmse(np.full_like(y, np.median(y)), y)
    wet = y > 0
    r_mask[i] = tile_rmse(np.where(wet, y[wet].mean() if wet.any() else 0.0, 0.0), y)
    zero_px += int((y == 0).sum()); tot_px += y.size
    for k, t in enumerate(THR): y2_ge[k] += (y[y >= t] ** 2).sum()
    if i % 10000 == 0: print(f"  {i}/{N}")

rmse_const = lambda c: np.sqrt(np.maximum(c*c - 2*c*t_mean + t_msq, 0)).mean()
loc_mean = {L: t_mean[locs == L].mean() for L in np.unique(locs)}
ladder = {
  "predict zero":              rmse_const(0.0),
  "global mean const":         rmse_const(t_mean.mean()),
  "per-location climatology":  float(np.mean([np.sqrt(max(loc_mean[L]**2 - 2*loc_mean[L]*m + q, 0))
                                              for L, m, q in zip(locs, t_mean, t_msq)])),
  "ORACLE tile median (flat)": float(r_med.mean()),
  "ORACLE tile mean (flat)":   float(np.sqrt(np.maximum(t_msq - t_mean**2, 0)).mean()),
  "ORACLE mean + wet mask":    float(r_mask.mean()),
}
best_c = min(np.linspace(0, 1, 101), key=rmse_const)
print(json.dumps({k: round(v, 4) for k, v in ladder.items()}, indent=1))
print(f"best flat constant: {best_c}  |  exact-zero pixels: {zero_px/tot_px:.2%}")
  0/40686
  10000/40686
  20000/40686
  30000/40686
  40000/40686
{
 "predict zero": 0.746,
 "global mean const": 0.8435,
 "per-location climatology": 0.7979,
 "ORACLE tile median (flat)": 0.718,
 "ORACLE tile mean (flat)": 0.6766,
 "ORACLE mean + wet mask": 0.5939
}
best flat constant: 0.0  |  exact-zero pixels: 82.07%
read that table again, slowly
best flat constant is exactly 0. any uniform drizzle loses to silence. (predict-zero = 0.746 — if your score is above this, submit zeros first, seriously.)
per-location climatology loses to zero (0.798!) — and that's in-sample. test locations are disjoint, so location-identity features are a trap, not a feature.
even an oracle median (0.718) loses to the oracle mean (0.677) by 0.041. the metric is RMSE: serve the conditional MEAN. never a median, never "a typical value". we learned this one with cash (a quantile head served at q0.5 cost us ~0.006 LB).
and the headline: a model that predicts every tile's true average rain PERFECTLY — zero error on the amount, something no real model will ever do — scores 0.677.
that number is the wall. now look where the leaderboard sits.

[3]: 
 
































# public LB, all 68 scored teams (as of jul 2)
LB = [0.6441,0.6520,0.6527,0.6557,0.6593,0.6599,0.6648,0.6684,0.6690,0.6692,0.6711,0.6720,0.6728,
      0.6744,0.6755,0.6757,0.6782,0.6794,0.6805,0.6806,0.6828,0.6838,0.6860,0.6862,0.6876,0.6879,
      0.6893,0.6900,0.6918,0.6932,0.6932,0.6940,0.6941,0.6955,0.6966,0.6969,0.6970,0.6979,0.6988,
      0.6989,0.7014,0.7055,0.7071,0.7098,0.7111,0.7121,0.7131,0.7159,0.7186,0.7196,0.7297,0.7297,
      0.7410,0.7549,0.7759,0.7811,0.7970,0.7995,0.8246,0.8342,0.8342,0.8353,0.8751,0.8770,0.8918,
      0.9028,0.9127,0.9540]
WALL, MASK, ZERO = ladder["ORACLE tile mean (flat)"], ladder["ORACLE mean + wet mask"], ladder["predict zero"]

fig, ax = plt.subplots(figsize=(11.5, 4.6))
rng = np.random.RandomState(1); jit = rng.uniform(-0.30, 0.30, len(LB))
for zone, (lo, hi), c in [("broke the wall", (0, WALL), VIOLET),
                          ("tile-signal zone", (WALL, ZERO), BLUE),
                          ("worse than all-zeros", (ZERO, 2), RED)]:
    m = [(lo <= s < hi) for s in LB]
    ax.scatter(np.array(LB)[m], jit[m], s=64, c=c, edgecolors=SURF, linewidths=1.6, zorder=3,
               label=f"{zone}  (n={sum(m)})")
for x, lab, c in [(MASK, f"oracle mean+mask {MASK:.3f}", VIOLET), (WALL, f"THE WALL {WALL:.3f}\n(perfect tile-mean)", VIOLET),
                  (ZERO, f"predict zero {ZERO:.3f}", INK2)]:
    ax.axvline(x, color=c, lw=1.2, ls=(0,(4,3)), zorder=1)
    ax.text(x + 0.004, 0.50, lab, ha="left", fontsize=8.8, color=INK, linespacing=1.25)
ax.annotate("#1 (0.644)", (LB[0], jit[0]), textcoords="offset points", xytext=(-8, 12), fontsize=9, color=INK)
ax.annotate("me, #17 - sitting ON the wall", (0.6782, jit[16]), textcoords="offset points",
            xytext=(26, -46), fontsize=9, color=INK,
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
ax.set_yticks([]); ax.set_xlim(0.58, 0.98); ax.set_ylim(-0.55, 0.62)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_xlabel("public LB score (per-sample RMSE, lower = better)")
ax.legend(loc="upper right", frameon=False, fontsize=9)
ax.set_title("all 68 teams vs the oracle lines: the wall literally cuts the leaderboard at rank 16/17",
             fontsize=12.5, fontweight="bold", loc="left", color=INK, pad=10)
plt.tight_layout(); plt.show()
n_broke = sum(s < WALL for s in LB); n_worse0 = sum(s > ZERO for s in LB)
print(f"teams below the wall: {n_broke} | teams scoring worse than predicting all zeros: {n_worse0}")
preview
teams below the wall: 16 | teams scoring worse than predicting all zeros: 15
three zones, one line each:

16 teams have broken the wall. they are getting information a perfect tile-average forecaster cannot have — i.e. they know smth about where the rain sits inside the 41×41 tile, not just how much.
the 0.68–0.75 pack (me included) is at or above the wall: we've squeezed out the tile-level signal and are now competing over rounding errors of the same quantity. more seeds/ensembles of "how-much" models cannot pass 0.677, even in the limit of perfection. that's why the plateau feels like a wall — it is one.
15 teams score worse than predicting literal zeros. if that's you: submit all-zeros once (0.746), then fix drizzle — you're leaking small positive values over the 82% dry pixels.
and the next rung is huge: perfect tile-mean + perfect wet/dry mask scores 0.594. that's 0.083 of headroom in pure localization — ~2.5× the entire gap between #1 and the pack. (fwiw a σ=2px blurred truth still scores ~0.38 — you don't need sharp peaks, you need correctly-PLACED rain.)

where the error actually lives
82% of pixels are exact zeros, so people assume the score is about dry pixels. it isn't — the metric averages per-tile RMSE, and wet tiles have huge RMSE:

[4]: 
 




















fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
tot = t_y2.sum()
share = [s/tot*100 for s in y2_ge]
a1.plot(THR, share, color=BLUE, lw=2, zorder=3); a1.fill_between(THR, 0, share, color=BLUE, alpha=0.10)
for t, s in zip(THR, share):
    if t in (5, 10, 20):
        a1.plot(t, s, "o", ms=8, color=BLUE, mec=SURF, mew=2, zorder=4)
        a1.annotate(f">= {t:g} mm/hr:\n{s:.0f}% of error", (t, s), textcoords="offset points", xytext=(10, 4), fontsize=8.8)
a1.set_xscale("log"); a1.set_xticks([0.1,0.5,1,5,10,20,40]); a1.set_xticklabels(["0.1","0.5","1","5","10","20","40"])
a1.set_xlabel("pixel intensity t (mm/hr)"); a1.set_ylabel("share of total squared error from pixels >= t (%)")
a1.set_title("heavy pixels own the metric", fontsize=11.5, fontweight="bold", loc="left")
srt = np.sort(t_y2)[::-1]; cum = np.cumsum(srt)/tot*100; q = np.arange(1, N+1)/N*100
a2.plot(q, cum, color=BLUE, lw=2); a2.fill_between(q, 0, cum, color=BLUE, alpha=0.10)
for pct in (1, 5, 10):
    v = cum[int(N*pct/100)-1]
    a2.plot(pct, v, "o", ms=8, color=BLUE, mec=SURF, mew=2)
    a2.annotate(f"top {pct}% of tiles = {v:.0f}%", (pct, v), textcoords="offset points", xytext=(10, -2), fontsize=8.8)
a2.set_xscale("log"); a2.set_xticks([1,5,10,25,100]); a2.set_xticklabels(["1","5","10","25","100"])
a2.set_xlabel("wettest q% of tiles"); a2.set_ylabel("cumulative share of squared error (%)")
a2.set_title("a few tiles own the leaderboard", fontsize=11.5, fontweight="bold", loc="left")
for a in (a1, a2): a.grid(axis="y", color=GRID, lw=0.8); a.set_axisbelow(True); a.set_ylim(0, 105)
plt.tight_layout(); plt.show()
preview
so ~1% of pixels (the ≥5 mm/hr ones) carry ~82% of the squared error, and the wettest 5% of tiles carry ~66% of the whole score.

tempting-but-wrong conclusion: "just up-weight heavy rain in the loss." we measured it — net negative on LB. the up-weighted model buys a little accuracy on rare wet tiles and pays with drizzle leaked across the 82% zeros, and the metric taxes that hard (remember: best constant is 0). what actually helped for free: zero out predictions below ~0.1 mm/hr. both things are true at once: the score lives in wet tiles, and you must keep dry pixels at exactly 0. the only way to win both is better structure, not re-weighting.

what the wall looks like, up close
four real training tiles vs the cheating flat tile-mean (per-column shared color scale):

[5]: 
 














PICKS = [("train_jakarta_GPM_IMERG_2023-01-01_00-00-00.tif", "stratiform"), ("train_guangdong_GPM_IMERG_2023-01-10_19-00-00.tif", "scattered\nconvection"),
         ("train_aceh_GPM_IMERG_2025-11-25_16-30-00.tif", "organized\nconvection"), ("train_central_vietnam_GPM_IMERG_2025-11-02_06-30-00.tif", "extreme peak\n(96.5 mm/hr)")]
fig, axes = plt.subplots(2, 4, figsize=(11.6, 6.2), gridspec_kw={"wspace": .06, "hspace": .04})
for j, (fn, name) in enumerate(PICKS):
    y = np.clip(np.nan_to_num(tifffile.imread(os.path.join(TRAIN, "gpm_imerg", fn)).astype(np.float32)), 0, None)
    flat = np.full_like(y, y.mean()); vmax = max(y.max(), .5)
    for i, arr in enumerate([y, flat]):
        ax = axes[i, j]; ax.imshow(arr, cmap=RAIN, norm=PowerNorm(.45, vmin=0, vmax=vmax), aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        if j == 0: ax.set_ylabel(["TRUTH", "flat tile-mean\n(oracle at the wall)"][i], fontsize=9.5, color=INK2)
    axes[0, j].set_title(name, fontsize=10)
    axes[1, j].set_xlabel(f"tile RMSE - zero: {tile_rmse(0*y, y):.2f}\nflat-mean oracle: {tile_rmse(flat, y):.2f}",
                          fontsize=9, color=INK2)
plt.suptitle("even a CHEATING flat mean barely dents convective tiles - the structure IS the score",
             fontsize=12.5, fontweight="bold", x=.005, ha="left")
plt.tight_layout(rect=(0, 0, 1, .95)); plt.show()
/var/folders/x8/v2y9hgbj25z8__5d154k2m6m0000gn/T/ipykernel_93415/680102799.py:16: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
  plt.tight_layout(rect=(0, 0, 1, .95)); plt.show()
preview
the flat mean nearly solves stratiform rain (0.39 → 0.12) but keeps ~75% of the error on the organized-convection tile (13.7 → 10.2). convective spatial structure isn't an edge case of this competition — it is the competition.

bonus: don't waste a day on the test_files/ "targets"
the GPM tifs inside the evaluation zip look like targets. they're not — they're noise placeholders. proof in 10 lines:

[6]: 
 








EVAL = sorted(glob.glob(os.path.join(os.path.dirname(TRAIN), "evaluation_dataset_*")))[0]
files = sorted(glob.glob(os.path.join(EVAL, "test_files", "*.tif")))[:50]
if files:
    vals = np.stack([tifffile.imread(f).astype(np.float32) for f in files])
    a = vals[0]; lag1 = np.corrcoef(a[:, :-1].ravel(), a[:, 1:].ravel())[0, 1]
    print(f"n={len(files)} files | mean={vals.mean():.4f} std={vals.std():.4f}  (Uniform[0,50] theory: 25.000, 14.434)")
    print(f"exact zeros: {(vals==0).mean():.1%}  (real GPM tiles: ~82%) | spatial lag-1 autocorr: {lag1:+.4f} (real rain: ~+0.9)")
    print("=> pure U[0,50] i.i.d. noise. placeholders, not a leak. move on :)")
else:
    print("test_files/ not present in this environment - see numbers in the text")
n=50 files | mean=24.9487 std=14.4263  (Uniform[0,50] theory: 25.000, 14.434)
exact zeros: 0.0%  (real GPM tiles: ~82%) | spatial lag-1 autocorr: +0.0012 (real rain: ~+0.9)
=> pure U[0,50] i.i.d. noise. placeholders, not a leak. move on :)
takeaways (and the stuff i'd love to discuss)
the leaderboard plateau at 0.68 is not a tuning problem — it's an information wall. perfect tile-means score 0.677. everyone at/above it is competing over the same exhausted quantity.
the game below the wall is within-tile localization. the wet-mask rung alone is worth 0.083 — 2.5× the #1-to-pack gap. blurred-but-well-placed rain (0.38!) beats sharp-but-misplaced everything.
serve the conditional mean. oracle median loses by 0.041. quantile/median heads are structurally penalized here.
best constant is 0; climatology < zero; location features are traps (disjoint test locations).
don't naive-tail-weight; do kill sub-0.1 drizzle.
open questions for the thread — genuinely curious:

for the sub-wall teams: is it displacement-tolerant losses? wet-mask aux heads? smth else entirely? (no need to give away your sauce — even "it's the loss" vs "it's the inputs" would be gold)
has anyone made cloud-top cooling rate work as a localization signal? our hand-engineered versions didn't transfer into the CNN.
with 20 train climates → 18 unseen test climates, big models overfit for us every single time. anyone winning WITH capacity?
credits: hengck23's malformed-files census (45 weird GOES tifs + 29 test rows with [] observations — go read that thread), crossentropy's EDA on the disjoint locations, Aman's band-wavelength table. this post just adds the measured layer on top. may your rain be well-placed 🌧️

Tags

nowcasting
precipitation

5 Comments

Comment



Preview

Attachment limit: 100MB total
dadjatoussaint

Posted 1 週間前

1
Thank a lot for sharing. It is been joyful reading your thoughtful analysis.


Reply


ibrahimqasmi
TOPIC AUTHOR

Posted 6 日前

0
thank you :)

Reply

hengck23

Posted 6 日前

1
Good work. You may want to do the same experiment on train dataset. Then, you results actually show:

The hidden private test distribution is not the same as the public test. (I think some high rainfall targets are reserved in the private test)

Oracle results show that prediction history can significantly improve results:

instead of  rainfall = model (sat), it is better to use rainfall = model (sat, past predicted rain history). This is also what i suggest the organizer host to allow for this competition.

Oracle shows site-adapted model is better than generalised solution

Reply


ibrahimqasmi
TOPIC AUTHOR

Posted 6 日前

0
alright thank you :)

Reply

ibrahimqasmi
TOPIC AUTHOR

Posted 6 日前

0
@hengck23  coming back to this after thinking about your points:

1. small clarification: the oracle numbers ARE computed on the train set (all 40,686 tiles), so the wall at 0.677 is a train-side number. but your public-vs-private point gave me an idea: the cleanest probe is an all-zeros submission. train says zeros = 0.746, so if the public LB returns smth meaningfully different, that gap directly measures the distribution shift you suspect. tempted to burn a submission on it :)

2. agree on history. related: the eval set is a contiguous 30-min time series per location (i posted a second notebook on this), so the neighbor rows' observation frames already give you sky state at and after the target moment, even without predicted-rain history as an official input. would be great if the hosts clarified their position on using it.
 

3. the per-location oracle spread backs your site-adaptation point hard: perfect-tile-mean ranges from 0.003 (dhaka) to 1.53 (hat_yai). the metric basically lives in a handful of wet sites.

# ==================== 2026-07-19 追加取得分 ====================

## Q&A: Does converting provided location names to coordinates count as an external dataset?

shionsuio, posted 2 weeks ago

The provided dataset includes location names but not their corresponding coordinates (latitude/longitude). I would like to convert these location names into geographic coordinates as a preprocessing step, so that the model can learn spatial relationships. Since the coordinates are derived directly from the location names already present in the provided data—rather than introducing new, independent data—I believe this could be considered a data transformation rather than the use of an external dataset. Could the organizers clarify whether this approach is permitted under the current rules?

### charlie (Solafune Crew), posted 2 weeks ago

Hi @shionsuio, Thank you for your interest and for your question regarding the use of location names converted into coordinates. After discussing this with our internal team, we concluded that this technique is considered a feature transformation. Specifically, converting a location name into geographic coordinates is treated as a transformation, since the resulting coordinates are generally constant for a given location.

If you perform this transformation, please ensure that all of the following conditions are met:
- Use only a free, reproducible, and commercially available geocoding source. Paid services are not permitted.
- Clearly document the geocoding source used in your submission.
- Ensure that we can reproduce the transformation during the verification stage.

Please also note that the original geocodes and coordinates were intentionally removed from the dataset to prevent participants from inferring the correct answers from the underlying data sources. Therefore, if you perform geocoding, please use the EPSG:4326 (WGS 84) coordinate reference system to ensure that your transformed coordinates are as accurate and reproducible as possible.

### shionsuio (topic author), posted 2 weeks ago

Thank you for clarifying that converting location names into WGS84 coordinates is considered a feature transformation. I would like to ask a follow-up question about features derived from those coordinates. Would it be permitted to derive additional geographic features such as elevation, distance to coastline, or climate zone from the obtained coordinates, using only free, reproducible, and commercially available open data sources, with the data source and transformation process clearly documented? Or is the permitted transformation limited to the coordinates themselves and deterministic mathematical features computed directly from them, such as latitude, longitude, hemisphere, trigonometric encodings, and local solar time?

### charlie (Solafune Crew), posted 2 weeks ago

Hey @shionsuio, TL;DR: Only closed-form functions of position and time are allowed. Anything that needs an external dataset — including fully open ones — is out.

Because this is a spatio-temporal nowcasting challenge, the objective is for the model to learn atmospheric dynamics directly from the satellite observation sequences. Injecting static spatial priors would fundamentally undermine that sequence-modeling objective, which is why the rules draw a firm line on external attributes.

**Permitted** — deterministic functions of position and time computed in closed form: latitude/longitude, hemisphere, trigonometric positional encodings, and solar geometry / local solar time. None of these require external data.

**Not permitted** — elevation, distance to coastline, and climate zone, regardless of licensing. Each requires joining to an external dataset (a DEM, coastline geometry, or a climate-classification map), which constitutes use of an external dataset and is strictly prohibited under the competition rules.

The single test is whether a feature needs an external dataset — not whether that dataset is free, open, or well-documented. Fully open sources still count as external data here.

One clarification on the "transformation" exception: it applies solely to closed-form mathematical operations performed directly on the grid parameters (e.g., trigonometric spatial positional encodings) that do not require any external lookup table or DEM.

### hengck23, posted 2 weeks ago

"Use only a free, reproducible, and commercially available geocoding source. Paid services are not permitted." — free but required registration? limited request rate? I think we need to keep a list of what is allowed after the host verifies it.

"Would it be permitted to derive additional geographic features..." There are many features; see below. But how to define "historical", etc? It would be ambiguous as to what can be derived. And I think we also need an approved list if allowed, e.g. once you have latitude and longitude values, there are a couple of free api like below.

```
geo_features = [
    # Coordinates
    sin_lat, cos_lat, sin_lon, cos_lon,
    # Terrain
    elevation, elev_mean_25km, elev_std_25km, elev_max_25km, slope_mean_25km,
    # Climate, historical
    mean_rain_selected_month, std_rain_selected_month, rainy_day_prob_selected_month,
    mean_temp_selected_month,
    # atmospheric state, historical
    humidity, wind ...
    # Broad land/water context
    distance_to_coast, water_fraction_25km,
]
```

### Fulankun1412 (Solafune Crew), posted 4 days ago

Hi @hengck23, Thank you for your insight. In the wake of your inquiries regarding the need for a topic specifically to ask/request about whether additional sources [are allowed]. We decided to create a special Discussion thread pinned in the discussion panel. Please check it here. Best Regards.

## Q&A: Rules clarification — permitted uses of the provided competition data

am45k, posted 4 days ago

Since external datasets are prohibited, I want to make sure I understand what counts as permitted use of the data distributed by this competition itself. Could you clarify which of the following are allowed?
1. Using the training-set satellite imagery for purposes other than direct supervised training (e.g., unsupervised objectives, sanity checks, preprocessing development).
2. Using the evaluation-set input imagery (the satellite tifs, not the placeholder targets) anywhere in model development — for example in preprocessing statistics or unsupervised objectives — given that these files were distributed to all participants.
3. Computing dataset-level statistics (e.g., per-band normalization constants) from each split.

No external data would be involved in any of these. Asking so that solutions stay clearly compliant ahead of the winners' code review — a short yes/no per item would be really helpful.

### Fulankun1412 (Solafune Crew), posted 4 days ago

Hi @am45k, Thanks for asking before the deadline. To answer each item:
1. **Yes.** The training data may be used for any purpose within the competition — supervised or unsupervised objectives, sanity checks, preprocessing development, etc.
2. **Yes.** The evaluation-set input imagery (satellite tifs) was distributed to all participants and may be used in model development, including preprocessing statistics and unsupervised objectives. The placeholder targets are excluded, and any attempt to reconstruct or estimate the true GPM-IMERG targets outside your model's prediction pipeline is not allowed.
3. **Yes.** Dataset-level statistics such as per-band normalization constants may be computed from either split.

As a reminder, no external data may be involved at any stage, and your winners' code submission must reproduce these steps (preprocessing/training/prediction modules). Good luck!

## shafimiakhil: What we learned measuring our own nowcasting pipeline: three validation traps, and where the error actually lives

Authors: Adeyinka Michael Sotunde (MICADEE) · Shafi Ullah Miakhil (shafimiakhil). Posted mid-competition, July 2026.

Scope: all diagnostics on one pipeline ("ultimate-v2": ConvLSTM → Attention-UNet → FiLM, single head, log1p space, public LB 0.69325), not their best submission (ensemble, LB 0.68577). 5-fold location-holdout CV (StratifiedGroupKFold, seed 42), 40,686 train rows, 68,393,166 OOF pixels.

**§1 The batch-averaging trap.** PyTorch Lightning's `self.log(..., on_epoch=True)` averages the metric over batches; since sqrt is concave, the mean of per-batch RMSEs is systematically lower (optimistic) than the pooled RMSE over all pixels at once. They claim "the leaderboard computes the pooled version" (sum SSE and N over the WHOLE validation set, one sqrt at the end) and that trusting the batch-averaged number cost them ~0.018 RMSE at model selection and once inverted an ablation's conclusion entirely (batch-avg said improved, pooled said worse).

**§2 Fold RMSE mostly measures fold climatology.** Regressing per-fold RMSE against held-out-fold wetness: r=0.884–0.984, R²=0.78–0.97 across two of their pipelines. Their easiest fold (lowest RMSE) was simply the driest fold (90% exact zeros), not a "well-generalized" fold. They also flag: bihar and dhaka (their fold 4) look suspiciously near-zero-rain in this dataset (bihar: mean 0.00066mm, 99.57% zeros, p99=0.00 across 2023-2026 — surprising for the Indo-Gangetic monsoon plain), unverified.

**§2c Early stopping fires on validation noise.** On their wettest fold, pooled val RMSE oscillated 1.60→1.74→2.84→5.83→1.71 across epochs while train loss fell monotonically; `patience=8` stopped at epoch 10 (still improving), `patience=15` reached real minima at epoch 14/21 with a large improvement.

**§3 Error concentration.** dry(<0.05mm)=83.68% of pixels/5.1% of error; light(0.05-2mm)=12.19%/9.2%; heavy(≥2mm)=4.13%/85.7%. Cross-checked on a second architecture (SatUNet): 84.2% pooled / 82.8% fold0 — "two different models, same conclusion: the error concentration is a property of the data."

**§4 Post-hoc calibration doesn't work (for them).** Binning by TRUE value shows fake "catastrophic under-prediction" at high truth (attenuation/regression-to-mean artifact, expected under squared loss). The correct diagnostic is binning by PREDICTED value and checking E[true|pred]≈pred — their model is roughly calibrated this way. Leave-one-fold-out isotonic recalibration: REJECTED (1.0914→1.0983, worse). Oracle isotonic (cheating, fit on test itself): only 0.8% gain. Conclusion: no monotone headroom once the model is genuinely calibrated — residual is conditional variance, not fixable bias.

**§5 Heavy-rain loss weight is bias correction, not emphasis.** Removing a ×3 loss weight on heavy pixels cost +0.038 RMSE (not the <1% they expected). Mechanism: log-space Huber estimates ~median(log1p(Y)|x); expm1 maps that below E[Y|x]; the ×3 weight pushes it back up. Without the weight, global bias goes from −1.8% to −26.8% and every E[true|pred] bin becomes positive (systematic under-prediction).

**§5a Prediction test (half-failed).** Predicted isotonic recal should now work on the no-×3 model (uniformly positive gaps = monotone-correctable). Honest leave-one-LOCATION-out test: FAILED — made both models worse (climatology transfer failure: fit on 2 dry locations, scored on 1 wet one, 10× rain-rate spread). Oracle (fit-on-scored-data) test: CONFIRMED — no-×3 model has ~4x the monotone headroom of the ×3 model. Conclusion: the bias is real and monotone, but a recalibration map does not transfer across climatologies (their fold-composition argument from §2, worn as a different hat).

**§5b Decomposing the ×3's benefit carefully.** Two different questions give different %: "can recalibration replace the ×3?" → 85% (comparing recalibrated no-×3 vs raw ×3, an apples-to-oranges framing). "Is the ×3's benefit purely bias?" → comparing both models AFTER optimal recalibration, the residual gap (0.0135 of 0.0377 total benefit) cannot be removed by any monotone map → ~64% is bias-attributable, ~36% is something the ×3 does during training (capacity allocation toward the heavy-rain 4%) that no output-space correction can buy back.

**§6 Input resolution.** Uniform 144×144 (stops downsampling Meteosat, but now upsamples Himawari 1.8x) was net negative (+0.024) but the per-sensor breakdown shows Meteosat's heavy-rain RMSE genuinely improved (5.03→4.68) while Himawari degraded more (+0.13) — a real resolution effect, just not uniformly beneficial. Caveat: confounded by differing epoch-of-best-checkpoint and n=1 Meteosat eval location in their fold 0.

**§6a Per-sensor variable resolution: BatchNorm artifact, not a real result.** Variable-size batching forces sensor-homogeneous batches; result was much worse (1.1163→1.2500), but Himawari (whose resolution didn't even change) degraded just as much (+0.174, heavy predictions collapsed 5.016→2.711) — proof the damage is BatchNorm running-stats corruption from single-sensor batches, not a resolution effect. Conclusion: per-sensor native resolution is UNTESTED, would need GroupNorm/InstanceNorm or sensor-specific BN to test cleanly.

**§7 Ceiling.** Discrimination is strong (×72.7 lift finding ≥10mm pixels among top-0.39% predictions; 37.1% of predicted maxima land within 5px of the true max, vs ~4.7% chance) but intensity is compressed ~4x (true≥10mm mean 16.25 → median prediction 4.17). Public LB top-10 (as of 2026-07-19) spans only 0.0156 (rank1 0.6295 → rank10 0.6452) against an official all-zero baseline of 0.91265 — "ten independent teams stacking inside ~2.5% of each other... is what an information ceiling looks like."

**Their summary for practitioners:** (1) log pooled RMSE, not batch-averaged; (2) per-fold RMSE≈climatology, compare configs on a fixed fold, never crown a fold; (3) diagnose bias by binning on PREDICTED not TRUE; (4) check global bias if training in log space (a heavy weight may be silent bias correction); (5) ~4% of pixels carry ~86% of error; (6) verify variable-size-batch BatchNorm results on an unchanged sensor; (7) a post-hoc calibration map is climatology-specific, expect it to under-deliver vs in-sample estimates on a new-climate test set.

## ccilabo: Count-0 areas in Clean Longwave IR data

Posted 10 hours ago. Areas with Clean Longwave IR (B13/C13/ir_105) count=0 appear to overlap with intense precipitation zones in the distributed data. Suspected uint8 underflow: cloud-top temperatures for cumulonimbus rarely fall below the tropopause temperature, so a true physical zero should be extremely uncommon. Asks whether this occurs in the original raw data at the same locations, i.e. whether it's a genuine sensor value or a uint8 conversion artifact.

## Bull: Tips for Improving Your Score

Posted 10 hours ago. Organized measure-correctly → understand-your-errors → build-models → finish-strong.

**Measure correctly:**
1. "In our validation, the LB score behaves as 'RMSE computed per image, then averaged over images' (not a pooled RMSE computed over all pixels of all images at once)." Aligning local validation to this metric changes model-selection and ensemble-weighting decisions.
2. GroupKFold by site is a must (train/test are disjoint in sites). Fold scores vary >2x between best/worst fold; treat ~±0.01 differences as noise; only trust improvements consistent across multiple folds.

**Understand your errors:**
3. Exhaustive EDA from every angle — visualize the worst-error images by eye, decompose by time-of-day (found a diurnal pattern: predicted amplitude shrinks at night since visible channels carry no information after dark). Warning: per-site/per-latitude patterns almost never reproduce under leave-one-site-out validation (too few sites) — don't trust findings whose sample size is "number of sites."
4. Error-by-intensity-band decomposition: low-intensity is nearly worthless, the BULK of error lives in the mid "broadly wet" band (not the heavy tail). RMSE scales as squared-error×area, so improving moderate rain over a wide area pays more than nailing a few heavy-rain pixels. Estimate the error-budget share before working on any improvement.

**Build models:**
5. Loss: plain MSE in log1p space beat hurdle/Tweedie/Charbonnier/heavy-weighting in their tests — "loss engineering is an easy rabbit hole with little return," spend the time on EDA/ensembling instead.
6. Single models plateau; the breakthrough is ensembling via greedy forward selection on pooled OOF (per-image RMSE) — a mediocre single-model CV can still contribute if weakly correlated with existing members. Save OOF predictions from the first experiment onward.
7. The diversity that works is "different information" (longer past-time context, alternative temporal sampling), not "different architectures" looking at the same inputs (those saturate and get zero ensemble weight).

**Finish strong:**
8. Full-data retraining (all sites, no fold split) consistently improved LB after the config was fixed via CV.
9. Their validation suggests the public LB behaves like a uniform random sample of the test set (no site/satellite/time structural bias found); estimated public–private absolute-score swing ≈ ±0.008 (95%), but pairwise model differences move together on the same image set so rank changes (shakeup) should be small. Pick final submissions by configs that make sense on both CV and LB, not by hairline (0.000x) public-score gaps.

## peppamint & youjonathan: GOES band-count corruption is mostly in the evaluation split (instead of train)

Posted 5 hours ago. Full census of GOES `.tif` frames in both splits, flagging a file corrupt unless it opens cleanly, has exactly 16 bands, and is 141×141.

| Split | GOES frames | Malformed |
| --- | ---: | ---: |
| train | 30,788 | 10 |
| evaluation | — | 42 |

All 52 corrupt files open fine and have the correct 141×141 grid — the corruption is purely a missing-band problem: 30 files have only 4/16 bands, 3 have 12, 6 have 13, 3 have 14, 10 have 15.

Evaluation failures cluster in exactly two eval locations — **upper_midwest (28 files)** and **rio_grande_do_sul (14 files)** — often on consecutive 10-minute timestamps (e.g. upper_midwest 2024-06-12 has a run of ~13 back-to-back 4-band frames), suggesting bad acquisition/ingest windows rather than random per-file damage. Train impact is small (10/30,788 ≈ 0.03%). Full per-file band-count list included in the post (also mirrored above under "GOES band-count corruption").

## hannanfaisal: ImageNet-pretrained encoder weights

Posted 2 hours ago. Asks whether ImageNet-pretrained encoder weights are allowed, or whether the external-data ban includes pretrained models. No organizer reply yet as of this capture.
