## Integrating Multi-Structure Covalent Docking with Machine-Learning Consensus Scoring Enhances Potency Ranking of Human Acetylcholinesterase Inhibitors \[[bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.09.29.679420v2)\]

### Overview

This code accompanies implements a machine-learning (ML) consensus method over multi-structure covalent docking scores to enhance the potency ranking of AChE inhibitors. It is readily extensible to analogous use-cases, and we hope to encourage its further development by offering this guide.

### Model & Approach

The ML conensus is implemented using a `scikit-learn` Random Forest regressor; see `scripts/get_ML_consensus.py` for exact architectural and hyperparameter optimization details. The primary measure of the method is the Spearman's rank correlation coefficient ($r_s$) its outputs achieve when compared to inhibitors' true potencies.

Two other baselines are also implemented for comparison on the basis of $r_s$:

1. single-structure methods (where the docking score associated with one of the five structures within our work is used to rank potency directly)
2. heuristic consensus methods (where the mean or minimum is taken over the five structures' docking scores instead of an ML aggregation)

### Dataset

For convenience, the datasets in consideration are stored as `pickle` files within `data/processed/`, pre-split into training and testing. Each file contains a list of two tuples, of the form
`[(X_train, y_train, meta_train), (X_test, y_test, meta_test)]`

`X_*` and `y_*` are `numpy` arrays storing the docking scores and potencies respectively. `meta_*` are `pandas` dataframes that supply contextualizing details (compound ID, cluster, class -- carbamate or organophosphate) on the compounds themselves.

### Training and testing the methods

After installing the conda environment (`conda env create -f environment.yml`), `python run.py` runs each benchmarking script in `scripts/`, including `compile_performance.py`, which calculates the $r_s$ values achieved by each method. The complete process will take fewer than 10 minutes on a consumer desktop.

### More on modifying the ML consensus

The following is the basic workflow of extending our work. While swapping out the Random Forest regressor for a different `scikit-learn` model (for example) is trivial -- it will require only a modification to `scripts/get_ML_consensus.py` -- we expect the more likely intent will be to alter the ML consensus method's input featurization. The below is a sketch of how to do so:

1. Produce new `pickle` files in the same tripartite-pair format as the existing ones.
2. Currently, each column of `X_*` corresponds to `PDB_IDS` as defined within `toolkit/config.py`. This helps `toolkit/data.py` identify how to subset `X_*` in the case not all structures are required as input. Alternative featurizations that no longer follow this logic (say, if you decide to use ligand physicochemical features) will need to adjust `get_split_input_data()` within `toolkit/data.py`. Multiple different sets of featurizations, which we currently define by `SCORE_TYPE` within `toolkit/config.py`, can already be handled.
3. To accommodate the changes to what each column of `X_*` now represents, modify the SHAP-related code block within `scripts/get_ML_consensus.py` accordingly.

### License

This project is licensed under the Apache 2.0 License -- see the [LICENSE](LICENSE) file for details.

### Citation

If this work is used in academic research, please cite:

_citation to be added after publishing_

### Contact

Please contact hfan2006 (at) gmail.com with questions or concerns.


