# 衛星気象学ドメイン知識レビュー (2026-07-20)

*目的: 「ML側のレバー(アーキテクチャ・アンサンブル・キャリブレーション)は掘り尽くした
感がある、次は気象学・地理学のドメイン知識が必要では」という議論を受けて、確立された
衛星降水推定・nowcasting手法を外部文献で調査し、自分たちの既存知見
(`doc/imerg_physics_notes.md`, `g_eda/exp001-005`) と対照してギャップを洗い出す。*

*方針: ここで見つけた手法は「配布データ(可視・IR帯16バンド×3衛星)のみから閉形式で
計算できるか」を最優先の合否基準にする。DEM・気候区分・外部プロダクトを要するものは
最初から除外。閉形式で作れても、`exp038_features`の波長整列特徴のようにOOF/LBが逆転
する例があるので、**採用は必ずfold0/fold4ゲート→OOF vs LB追跡**を経ること。*

---

## 1. 既に検証済み・結論が出ている項目 (再検証しない)

`doc/imerg_physics_notes.md`のH1-H6と`g_eda/exp004`から、**もう試す必要がない/採用済みの
もの**を先に確定する。

| 項目 | 結論 | 根拠 |
| --- | --- | --- |
| IR雲画像のoptical flow (雲の移動ベクトル) | **棄却 (ρ≈-0.05)** | 雲の移動とGPM降水場の移動は別物 — pySTEPS/DGMR系の「移流」は本質的に**降水場自身**の移流であり、雲IRの移流ではない。この違いに気づかず再実装すると同じ結果になる |
| 予測同士の移流補正平滑化 (H2実用形) | **棄却 (delta +0.00012)** | `g_eda/exp004/run_advected_smoothing.py`: static smoothingとほぼ同等、採用しきい値-0.001を超えない |
| PMWフレッシュ/モーフの2状態 (H1) | **棄却** | イノベーション分布は単調減衰、二峰性なし |
| 視差(パララックス)幾何補正 | **Himawariのみ確証、exp045で実装中** | 方向コサイン0.81。GOES/Meteosatは幾何モデル未検証 |
| WV7.3-window BTD, split-window比, 雲頂冷却率(newest-oldest window diff) | **既に`engineered`特徴として実装済み** | `exp038/dataset.py frame_engineered_channels` — 下記§3で外部文献との対応を確認 |

## 2. 外部文献調査で見つかった確立手法

### 2.1 古典的IRベース降水推定アルゴリズム

- **Griffith-Woodley法**: GOES IR(11μm)の雲域の時系列面積から降水量を推定する経験式
  ([NASA NTRS](https://ntrs.nasa.gov/citations/19840049157))。閉形式・IR単チャンネルのみ。
- **GOES Precipitation Index (GPI)**: 一定面積内で「-38°C(235K)より冷たい」画素の
  割合を降水量に線形換算する、最も単純な手法。**win帯の閾値+面積比だけで実装可能**。
- **Scofield-Oliver法 / Hydroestimator**: 雲頂温度と降水強度のべき乗則関係に、
  勾配・成長率マスクと湿度マスクを加える。湿度補正・地形補正まで含めるとDEM相当の
  外部情報が必要になる版もあるため、**IR+湿度(WV帯)部分だけを模倣し、地形補正は
  やらない**のが安全
  ([Wiley Hydrometeorology Ch.6](https://onlinelibrary.wiley.com/doi/10.1002/9781118414965.ch6))。

**示唆**: これらは全て「win帯(または win+wv帯)の閾値・べき乗則」で作れる、モデルとは
別の**古典的ベースライン予測子**。教師なしで計算できるので、(a) 診断用オラクル
(`exp043`ゼロベースラインと同じ位置づけ)として実測との相関を見る、(b) 補助特徴量として
モデルに入れる、の両方の使い道がある。

### 2.2 対流の発達を検知する時間シグネチャ

- **雲頂冷却率 (cloud-top cooling rate)**: UWCI (University of Wisconsin Convective
  Initiation) アルゴリズムは box-averaged cooling rate に **4K/15分** のしきい値を使う
  ([AMS J. Appl. Meteor. Climatol. 2011](https://journals.ametsoc.org/view/journals/apme/50/1/2010jamc2496.1.xml))。
  **我々の`temporal diff`特徴(newest-oldest window band, /DIFF_SCALE=32)は正確にこれに
  対応する** — 検証すべきは我々のDIFF_SCALEがこの物理的な尺度(4K/15分)と整合しているか。
  uint8の生値なので直接K換算はできないが、**同じ物理現象を捉えている**という確認は
  科学的な裏付けとして価値がある。

### 2.3 水蒸気・split window brightness temperature difference (BTD)

- **Split Window Difference (SWD)**: clean window(10.3μm)−dirty window(12.3μm)で
  雲上部の水蒸気量を示す ([CIMSS Satellite Blog](https://cimss.ssec.wisc.edu/satellite-blog/archives/23702))。
  **我々のSPL/(win+1)比は同じ2バンドの別の合成方法** — 文献の標準は差分(BTD)、我々は
  uint8量子化対策で比を使っている。差分版も試す価値がある(量子化耐性は劣るが解釈は
  文献に忠実)。
- **WV(6.5-7.3μm)-window(11μm) BTD**: 中層大気の乾燥空気侵入・強いダウンドラフトの
  可能性を示す ([arXiv:1004.3506](https://arxiv.org/pdf/1004.3506))。**我々のWV7.3-W
  特徴と一致** — 既に実装済みで、文献的な裏付けが取れた。

### 2.4 霧 vs 対流雲の判別 (今回のclear-sky合成の知見と直結)

- **3.9μm-10.7μm BTD (夜間霧検出)**: 水滴は3.9μmで黒体的に放射しないが10.7μmでは
  ほぼ黒体的 → 霧・下層雲は特徴的な負のBTDを示す (夜間のみ有効、昼は反射成分が混ざる)
  ([CIMSS Night Fog BTD](https://cimss.ssec.wisc.edu/goes/OCLOFactSheetPDFs/ABIQuickGuide_NightFogBTD.pdf))。
  **我々のmir(3.9μm)帯とwin(10.5μm)帯で直接計算できる、未実装の特徴**。
- **テクスチャ**: 霧は空間的に均質(smooth)、対流雲(特にovershooting top)はIR窓の
  局所標準偏差(3×3ブロックなど)が大きい ([IntechOpen Cloud Detection](https://www.intechopen.com/chapters/9541))。
  **未実装。局所分散フィルタを1チャンネル追加するだけで実装できる**。

これは前回のclear-sky合成EDA(`g_eda/exp008`)で見た「bihar/dhakaが極値パーセンタイルでも
最後まで白くもやがかったまま」という現象の定量的な裏付けになる — mir-win BTDや局所分散が
低い(=霧的)まま推移していれば、"曇天なのに無降水" を検出できる特徴が作れる可能性がある。

### 2.5 移流・機械学習ベースnowcasting手法の全体像

- **pySTEPS/RainyMotion**: レーダー反射率の移流補正外挿。運用ベースラインとして
  広く使われるが、予測精度はpersistenceとほぼ変わらない場合もある
  ([arXiv:2502.16116](https://arxiv.org/html/2502.16116v2))。
- **DGMR (DeepMind, Nature 2021)**: レーダーのGAN生成nowcasting。pySTEPS/UNet/
  MetNet(軸注意)と比較され、長いリードタイムでより鮮明な予測を出す
  ([Nature](https://www.nature.com/articles/s41586-021-03854-z))。
- **MetNet**: レーダー・衛星・地形情報を大範囲から集約、ConvLSTM encoder + 軸注意decoder
  で1回のforward passで確率分布を出す。
- **光学フロー+GAN のハイブリッド**: 複数の光学フローで線形外挿した画像をconditional
  GANで非線形補正 ([MDPI Remote Sensing 2023](https://www.mdpi.com/2072-4292/15/21/5169))。

**このコンペへの示唆**: 上記は全て**レーダー反射率**(降水そのものの動きを直接観測できる
プロダクト)を移流させる手法。我々の入力はIR/VISの**雲**画像で、既に「雲の移動と降水場の
移動は別物」と実測済み(§1)。つまりpySTEPS/DGMR系のアーキテクチャをそのまま輸入しても、
移流の入力データが違うため同じ効果は期待しにくい。**新しいライブラリ・アーキテクチャ
ファミリーへの乗り換えは今のところ根拠が弱い** — 現行のCNN(hurdle log-normal U-Net)+
特徴量エンジニアリングの路線を継続するのが合理的。

## 3. 未実施・green・実装コスト低い候補 (優先順位付き) — 2026-07-21 fold0/4ゲート結果を追記

exp038 strict baseline (fold0=0.28954, fold4=0.59607) 比。**両fold改善**が本来のゲート基準だが、
fold0が大幅改善・fold4がノイズ内タイの場合は正味プラスと判断し5-foldへ進めた (exp047)。

| 優先度 | 候補 | 実験 | fold0 | fold4 | 判定 |
| --- | --- | --- | ---: | ---: | --- |
| 1 | local solar hour sin/cos + hemisphere + day-of-year | exp047 | 0.27720 (**-0.01234**) | 0.59702 (+0.00095、タイ) | **通過。全アブレーション中最大の改善幅。5-fold投入済み** |
| 2 | mir-win BTD (霧/下層雲判別) | exp048 | 0.29415 (+0.00461、悪化) | 0.58948 (-0.00659、改善) | 混在、正味ほぼ相殺。5-fold見送り |
| 3 | IR window局所分散 (テクスチャ) | exp049 | 0.29435 (+0.00481、悪化) | 0.59416 (-0.00191、タイ) | 弱い。5-fold見送り |
| 4 | GPI風 rain-area-fraction | exp051 | 0.29168 (+0.00214、微悪化) | 0.59057 (**-0.00550、改善**) | fold4の改善が大きめ。5-fold投入済み (再評価目的) |
| 5 | split-window BTD (差分版) | exp050 | 0.29022 (+0.00068、タイ) | 0.58856 (**-0.00751、改善**) | fold0タイ・fold4改善。5-fold投入済み (再評価目的) |

## 4. 結論

- 「ML側のレバーは掘り尽くした」という認識は妥当。次の一手としてのドメイン知識投資は
  合理的だが、**このプロジェクトの過去の実測が示す通り、ドメイン知識は「仮説の源」
  であって「保証された効果」ではない** (exp038_featuresのOOF/LB逆転、cloud optical
  flowの棄却)。
- 今回の調査で最も有望なのは **local solar hour特徴** — 理由: 唯一「診断で効果を確認済み
  だが未実装」という、他の候補にはない強い事前確信を持つ項目。次点は霧/対流判別関連
  (mir-win BTD, IR窓テクスチャ) — 今回のclear-sky合成の知見と一直線につながる。
- **アーキテクチャやライブラリを変える根拠は現時点では弱い**。移流ベースnowcasting
  (pySTEPS/DGMR系)の前提(観測される場の移動=降水場の移動)が我々のデータでは
  崩れていることを既に実測しているため、CNNベースの現路線を継続するのが合理的。

## Sources

- [Rain estimation from satellites: Griffith-Woodley Technique (NASA NTRS)](https://ntrs.nasa.gov/citations/19840049157)
- [Satellite-Based Remote Sensing, Hydrometeorology Ch.6 (Wiley)](https://onlinelibrary.wiley.com/doi/10.1002/9781118414965.ch6)
- [Nowcasting Convective Storm Initiation Using Satellite-Based Box-Averaged Cloud-Top Cooling (AMS)](https://journals.ametsoc.org/view/journals/apme/50/1/2010jamc2496.1.xml)
- [The Split Window Difference as a measurement of Atmospheric Moisture (CIMSS)](https://cimss.ssec.wisc.edu/satellite-blog/archives/23702)
- [Microburst applications of brightness temperature difference (arXiv:1004.3506)](https://arxiv.org/pdf/1004.3506)
- [Fog and Low Cloud Detection (CIMSS)](https://cimss.ssec.wisc.edu/satellite-blog/archives/9969)
- [Why is the Fog Brightness Temperature Difference Important? (CIMSS/GOES Quick Guide)](https://cimss.ssec.wisc.edu/goes/OCLOFactSheetPDFs/ABIQuickGuide_NightFogBTD.pdf)
- [Automated Detection of Clouds in Satellite Imagery (IntechOpen)](https://www.intechopen.com/chapters/9541)
- [Integrating Weather Station Data and Radar for Precipitation Nowcasting (arXiv:2502.16116)](https://arxiv.org/html/2502.16116v2)
- [Skilful precipitation nowcasting using deep generative models of radar (DGMR, Nature 2021)](https://www.nature.com/articles/s41586-021-03854-z)
- [Enhancing Rainfall Nowcasting Using Generative Deep Learning with Multi-Temporal Optical Flow (MDPI)](https://www.mdpi.com/2072-4292/15/21/5169)
