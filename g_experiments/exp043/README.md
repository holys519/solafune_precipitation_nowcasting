# exp043: All-Zero Eval Baseline (診断専用)

モデルなし・学習なし。evalの29,090タイル全てにゼロを予測するだけの提出。目的はスコア改善
ではなく、**「なぜOOF(0.59〜0.61)と実LB(0.68〜0.69)にギャップがあるか」**という仮説の
直接検証。ディスカッション(ibrahimqasmi, `discussion/discussion_all.md`)で提案されていた
「all-zero提出でtrain側の壁(0.746, 40,686タイル全件でのコミュニティ計測値)との差を測る」
手法をそのまま実行する。

## 判断基準

- eval-zeroがtrain-zero(0.746)に近い → OOF→LBギャップは主にモデルの地点汎化失敗
- eval-zeroがtrain-zeroより明確に悪い(例: 0.80+) → evalサンプル自体がtrainよりwetな
  regime構成であることを、モデル抜きの数値で独立に裏付ける (E-3/E-4のregime-shift仮説の
  直接検証)

注記: 過去の`EXPERIMENT_REPORT.md`にある「zero baseline RMSE 0.962228」はexp001の
3000行ホールドアウトでの値であり、コミュニティの0.746(train全40,686タイル)とは
サンプルサイズが異なるため直接比較できない。

## Run

```bash
cd g_experiments/exp043
sbatch singularity_run.sh
```

出力: `outputs/submissions/exp043_zero_baseline.zip` (green, 配布データもモデルも使わない
純粋な定数予測なので規約上最も安全)
