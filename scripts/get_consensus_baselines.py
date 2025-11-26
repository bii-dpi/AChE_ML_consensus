import os
import numpy as np
import pandas as pd

from toolkit import config, data


def save_predictions(consensus_type, score_type):
    _, (X_test, y_test, meta_test) = data.get_split_input_data(
        score_types=score_type
    )

    predictions = (
        np.min(X_test, axis=1)
        if consensus_type == "min"
        else np.mean(X_test, axis=1)
    )
    method = f"{consensus_type}_{score_type}"

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
    for consensus_type in ("min", "mean"):
        for score_type in config.SCORE_TYPES:
            save_predictions(consensus_type, score_type)


if __name__ == "__main__":
    main()

