"""Extract best_score/best_rmse from all checkpoint files across g_model, g_model2, g_model3, l_model."""
import torch
import glob
import os
from collections import defaultdict


def extract_scores(base_dir: str) -> None:
    """Scan model directories and print a summary of best scores per experiment."""
    scores = defaultdict(list)

    for gdir in ['g_model', 'g_model2', 'g_model3', 'l_model']:
        full_dir = os.path.join(base_dir, gdir)
        if not os.path.isdir(full_dir):
            continue
        checkpoints = sorted(
            glob.glob(os.path.join(full_dir, '**', '*.pth'), recursive=True)
            + glob.glob(os.path.join(full_dir, '**', '*.pt'), recursive=True)
        )
        for f in checkpoints:
            try:
                ckpt = torch.load(f, map_location='cpu', weights_only=False)
                name = os.path.splitext(os.path.basename(f))[0].replace('_best', '')
                parts = name.rsplit('_fold', 1)
                if len(parts) == 2 and parts[1].isdigit():
                    exp = parts[0]
                    fold = int(parts[1])
                else:
                    # exp001/exp002-style checkpoints live at <model_dir>/<exp_name>/best_model.pt
                    exp = os.path.basename(os.path.dirname(f)) or name
                    fold = -1
                score = ckpt.get('best_score', ckpt.get('best_rmse', 0))
                scores[f"{gdir}/{exp}"].append((fold, float(score)))
            except Exception as e:
                print(f"Error loading {os.path.basename(f)}: {e}")

    print("=" * 90)
    print(f"{'Experiment':<45} {'Folds':<6} {'Mean':<8} {'Min':<8} {'Max':<8} Per-Fold")
    print("-" * 90)
    for key in sorted(scores.keys()):
        folds = sorted(scores[key], key=lambda x: x[0])
        vals = [s for _, s in folds]
        mean_s = sum(vals) / len(vals)
        fold_str = ' '.join([f'F{f}={s:.4f}' for f, s in folds])
        print(f"{key:<45} {len(vals):<6} {mean_s:<8.5f} {min(vals):<8.5f} {max(vals):<8.5f} {fold_str}")
    print("=" * 90)


if __name__ == '__main__':
    # Update this path to your project's base directory on the cluster
    base = os.path.dirname(os.path.abspath(__file__))
    extract_scores(base)
