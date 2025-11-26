import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from toolkit import config, data


def spearman_scorer(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else -1


def generate_weight_grid(resolution=0.01):
    weights = []
    for w in np.arange(0, 1 + resolution, resolution):
        w1 = round(w, 2)
        w2 = round(1 - w, 2)
        weights.append((w1, w2))
    return weights


def save_predictions(pdb_id):
    (X_train, y_train, _), (X_test, y_test, meta_test) = data.get_split_input_data(
        structure=pdb_id,
        score_types=["pre_reaction_score", "post_reaction_score"]
    )

    weight_grid = generate_weight_grid(resolution=0.01)
    kf = KFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )

    fold_weights = []
    fold_spearmans = []

    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train[train_idx], X_train[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]

        best_spearman_tr = -np.inf
        best_weights = None

        for w1, w2 in weight_grid:
            preds = w1 * X_tr[:, 0] + w2 * X_tr[:, 1]
            score = spearman_scorer(y_tr, preds)
            if score > best_spearman_tr:
                best_spearman_tr = score
                best_weights = (w1, w2)

        preds = best_weights[0] * X_val[:, 0] + \
         best_weights[1] * X_val[:, 1]
        best_spearman_val = spearman_scorer(y_val, preds)

        fold_weights.append(best_weights)
        fold_spearmans.append(best_spearman_val)

    # Normalize fold Spearman scores to use as weights
    total_corr = sum(fold_spearmans)
    normalized_wweights = [s / total_corr for s in fold_spearmans]

    # Compute final weighted average of weight pairs
    final_w1 = sum(ww * pair[0] for ww, pair in zip(normalized_wweights, fold_weights))
    final_w2 = sum(ww * pair[1] for ww, pair in zip(normalized_wweights, fold_weights))

    weights = np.array([final_w1, final_w2])
    print(f"{pdb_id} reweighted weights: {weights.round(3)}")

    predictions = np.dot(X_test, weights)
    method = f"{pdb_id}_reweighted"

    output_df = meta_test.copy()
    output_df["logIC50"] = y_test
    output_df["prediction"] = predictions
    output_df["method"] = method

    output_df = output_df[
        ["method", "compound", "class", "cluster", "logIC50", "prediction"]
    ]

    output_path = os.path.join(config.PREDICTIONS_DIR, f"{method}.csv")
    output_df.to_csv(output_path, index=False)


def main():
    for pdb_id in config.PDB_IDS:
        save_predictions(pdb_id)


if __name__ == "__main__":
    main()
