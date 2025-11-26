import os
import pandas as pd
from scipy.stats import spearmanr

from toolkit import config


OUTPUT_PATH = os.path.join(config.PERFORMANCE_DIR, "correlations.csv")


def summarize_file(filepath):
    df = pd.read_csv(filepath)
    method = df["method"].iloc[0]

    rho_overall = spearmanr(df["logIC50"], df["prediction"]).correlation
    rho_carbamate = spearmanr(
        df[df["class"] == "carbamate"]["logIC50"],
        df[df["class"] == "carbamate"]["prediction"]
    ).correlation
    rho_phosphate = spearmanr(
        df[df["class"] == "phosphate"]["logIC50"],
        df[df["class"] == "phosphate"]["prediction"]
    ).correlation

    cluster_rhos = {}
    for cluster_id in range(1, 13):
        cluster_df = df[df["cluster"] == cluster_id]
        rho = spearmanr(cluster_df["logIC50"], cluster_df["prediction"]).correlation
        cluster_rhos[f"rho_cluster_{cluster_id}"] = rho

    rho_clusters = sum(cluster_rhos.values()) / len(cluster_rhos)

    return {
        "method": method,
        "rho_overall": rho_overall,
        "rho_carbamate": rho_carbamate,
        "rho_phosphate": rho_phosphate,
        "rho_clusters": rho_clusters,
        **cluster_rhos
    }


def main():
    summaries = []
    for fname in os.listdir(config.PREDICTIONS_DIR):
        if fname.endswith(".csv"):
            fpath = os.path.join(config.PREDICTIONS_DIR, fname)
            summary = summarize_file(fpath)
            summaries.append(summary)

    df = pd.DataFrame(summaries)
    ordered_columns = [
        "method",
        "rho_overall",
        "rho_carbamate",
        "rho_phosphate",
        "rho_clusters",
    ] + [f"rho_cluster_{i}" for i in range(1, 13)]

    df = df[ordered_columns]
    df.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()

