import os
import subprocess


SCRIPTS = [
    "get_single_baselines",
    "get_consensus_baselines",
    "get_reweighted_covdock",
    "get_ML_consensus",
    "compile_performance",
]


def main():
    os.system("rm outputs/predictions/*")
    os.system("rm -f data/processed/compounds.pkl")

    for script in SCRIPTS:
        print(f"Running: scripts.{script}")
        subprocess.run(
            ["python", "-m", f"scripts.{script}"],
            check=True
        )


if __name__ == "__main__":
    main()

