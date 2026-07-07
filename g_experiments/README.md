# g_experiments

GPU/HPC向けの本番実験を格納します。ローカルで検証した内容を、同じ実験番号でこちらに反映します。

## Standard Layout

```text
exp001/
├── config.yaml
├── train.py
├── model.py
├── dataset.py
├── losses.py
├── metrics.py
├── inference.py
├── make_submission.py
├── singularity_run.sh
├── python_run.sh
└── README.md
```

## Conventions

- 実験番号は `exp001`, `exp002` の3桁連番。
- 各実験は自己完結させ、再現に必要な設定を `config.yaml` に残す。
- 学習済み重みは `../g_model/` に保存し、Gitには含めない。
- 提出zipは `../outputs/submissions/` に作成する。
- 外部データは使用しない。
