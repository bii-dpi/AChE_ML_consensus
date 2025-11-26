import os
import pandas as pd

from toolkit import data, config


def save_predictions(pdb_id, score_type):
    _, (X_test, y_test, meta_test) = data.get_split_input_data(
        structure=pdb_id,
        score_types=score_type
    )

    predictions = X_test[:, 0]
    method = f"{pdb_id}_{score_type}"

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
        for score_type in config.SCORE_TYPES:
            save_predictions(pdb_id, score_type)


if __name__ == "__main__":
    main()

