# l_eda/exp005: robust OOF validation toolkit (I-002)

`doc/plan/round5_experiment_plan_2026-07-16.md`のE-3/E-4の直接の後継。2026-07-24セッションの
3件の提出 (`exp050_sigmafixed`, `exp047_sigmafixed`, `exp055`) がいずれも「OOFは改善したのに
LBは転移しない、あるいは悪化した」ことを受けて、**提出前にその失敗パターンを検出するツール**
として作った。

## 背景: なぜ点推定のOOFだけでは足りないか

- train locationは20個しかなく、5-fold GroupKFoldでは1foldあたり2〜5locationしかない
  (fold0はわずか2location)。`l_eda/exp004` (E-4) が示した通り、fold間のRMSE分散の大半は
  モデルの質ではなく「どのlocationがどのfoldに入ったか」というregime構成で説明できる。
  つまり **0.001〜0.005程度のOOF差は、ほとんどの場合locationの組み合わせのノイズ** であり、
  モデルの改善を意味しない。
- `g_eda/exp011`のブレンド重み探索は、重みの当てはめと評価を同じOOFタイルに対して行っていた
  (in-sample fit)。`exp055`はこれによりOOFで-0.0087の改善に見えたものが実測では+0.0006の
  悪化に反転した。
- `exp047_sigmafixed`はOOFで-0.00081改善したのに実測LBは+0.0135の致命的な悪化。原因は
  location-identityに紐づくshortcut(hemisphere×衛星one-hotによる訓練地域気候の暗記)が
  疑われている — evalのlocationは訓練と非重複なので、こうした「特定の場所で良くなった」だけの
  改善は原理的に転移しない。

## ツール構成

1. **`bootstrap_ci.py`** — location-cluster bootstrap CI。20locationを1リサンプル単位として
   復元抽出を繰り返し、candidate−baselineのOOF差の分布を作る。80%区間が0をまたぐ場合、
   その差は「たまたまどのlocationを引いたか」のノイズと区別できない。
2. **`leakage_audit.py`** — gain-concentration監査。location別の改善幅を計算し、上位1〜3
   locationが総改善量の何割を占めるかを見る。集中している場合、`exp047_sigmafixed`のような
   location-identity shortcutの可能性を疑う(物理的に汎化する改善は複数locationに分散するはず)。
   衛星別の内訳も併記する。
3. **`submission_gate.py`** — 上記2つを統合し、GO/HOLD/NO-GOを機械的に判定する:
   - `NO-GO`: |Δ| が `l_eda/exp003`のノイズ床(0.004)未満、または80%CIが0を含む
   - `HOLD`: ノイズ床とCIは通過したが、改善が上位2locationに60%超集中している
     (シナリオ的にありうる = 要目視確認してからGO)
   - `GO`: 上記すべてクリア

## 使い方

```bash
cd l_eda/exp005
python3 submission_gate.py --baseline exp038_sigmafixed --candidate <候補expの analysis名>
```

入力は各実験の`analyze_oof.py`がすでに書き出している`outputs/analysis/{exp}/oof_sample_metrics.csv`
のみ。学習・推論は一切行わない (l_eda/exp003/exp004と同じ設計)。ブレンド(複数モデル合成)を
候補にしたい場合は、先に`g_eda/exp011/nested_blend.py`を実行して`{name}_nested_blend`という
analysis名で疑似`oof_sample_metrics.csv`を書き出させてから、それを`--candidate`に渡す。

## 遡及検証 (2026-07-24): 直近3件の失敗を実際に検出できるか

| 候補 (vs champion exp038_sigmafixed) | 実測LB | このツールの判定 | 根拠 |
| --- | --- | --- | --- |
| exp050_sigmafixed | +0.00038 (ほぼ横ばい) | **NO-GO** | Δ-0.00185は ノイズ床未満。80%CI [-0.00745, +0.00345] は0を含む。加えてtop2 location (guangdong/jamaica) が改善の69%を占め集中 |
| exp047_sigmafixed | **+0.01350 (致命的悪化)** | **NO-GO** | Δ-0.00081はノイズ床未満。80%CI [-0.00651, +0.00517] は0を含む。top2 location (guangdong/ecuador) が改善の74%を占め強く集中 — 実際に起きた壊滅的な悪化と整合 |

いずれも実際の提出前にこのツールを走らせていれば「NO-GO」と出ており、提出予算(残り約10日、
締切2026-08-03目安)を無駄にしていなかった。両ケースで`guangdong`が改善上位1位に共通して
現れている点は、モデル改善ではなく`guangdong`固有のデータ特性(何らかのlocation-specific
artifact)を拾っている可能性を示唆しており、追加調査に値する。

`exp055`(ブレンドのOOF/LB逆転)は`g_eda/exp011/nested_blend.py`側の遡及検証を参照。

## 既知の限界

- location数が20と少ないため、bootstrap CIはそれ自体広い(統計的検出力が低い) — これは
  バグではなく実際の不確実性を正直に表現した結果。「CIが広いから使えない」のではなく
  「この程度の証拠しかない」という事実を可視化することが目的。
- `leakage_audit.py`は緯度・半球などの外部地理情報を使わない、純粋に統計的な集中度診断。
  `exp047`のような特定メカニズムの確証はできない(それは別途手動診断が必要)が、
  「疑うべきshortcutの兆候」を機械的にスクリーニングできる。
