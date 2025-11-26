import os
import numpy as np
import pandas as pd
import optuna
import joblib
import shap

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.stats import spearmanr
from sklearn.neighbors import KernelDensity

from optuna.samplers import TPESampler
from toolkit import config, data

optuna.logging.set_verbosity(optuna.logging.WARNING)


def spearman_scorer(y_true, y_pred):
    corr, _ = spearmanr(y_true, y_pred)
    return corr if not np.isnan(corr) else -1


def get_kde_weights(X, bandwidth):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(X)
    log_density = kde.score_samples(X)
    density = np.exp(log_density)
    weights = 1.0 / (density + 1e-8)
    return weights / np.mean(weights)


def objective(trial, X, y):
    bandwidth = trial.suggest_float("bandwidth", 0.1, 2.0, log=True)

    model = RandomForestRegressor(
        n_estimators=trial.suggest_int("n_estimators", 50, 300),
        max_depth=trial.suggest_int("max_depth", 2, 20),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )

    sample_weight = get_kde_weights(X, bandwidth)

    scores = []
    kf = KFold(
        n_splits=config.NUM_FOLDS,
        shuffle=True,
        random_state=config.RANDOM_SEED
    )
    for train_idx, val_idx in kf.split(X):
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]
        w_train_cv = sample_weight[train_idx]

        model.fit(
            X_train_cv,
            y_train_cv,
            sample_weight=w_train_cv
        )
        y_pred = model.predict(X_val_cv)
        scores.append(spearman_scorer(y_val_cv, y_pred))

    return np.mean(scores)


def save_predictions(score_types):
    (
        X_train,
        y_train,
        meta_train
    ), (
        X_test,
        y_test,
        meta_test
    ) = data.get_split_input_data(score_types=score_types)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=config.RANDOM_SEED)
    )
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=config.N_TRIALS,
        show_progress_bar=True
    )
    best_params = study.best_params

    model = RandomForestRegressor(
        n_estimators=best_params["n_estimators"],
        max_depth=best_params["max_depth"],
        min_samples_split=best_params["min_samples_split"],
        min_samples_leaf=best_params["min_samples_leaf"],
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )

    sample_weight = get_kde_weights(X_train, best_params["bandwidth"])
    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weight
    )

    predictions = model.predict(X_test)

    method = (
        "ML_both" if isinstance(score_types, list)
        else f"ML_{score_types}"
    )
    
    cv_output_path = os.path.join(config.SUPP_DIR, f"cv_{method}.csv")
    study.trials_dataframe().to_csv(cv_output_path, index=False)

    output_df = meta_test.copy()
    output_df["logIC50"] = y_test
    output_df["prediction"] = predictions
    output_df["method"] = method

    output_df = output_df[
        ["method", "compound", "class", "cluster", "logIC50", "prediction"]
    ]

    output_path = os.path.join(config.PREDICTIONS_DIR, f"{method}.csv")
    output_df.to_csv(output_path, index=False)

    model_path = os.path.join(config.MODELS_DIR, f"{method}.pkl")
    joblib.dump(model, model_path)

    config_path = os.path.join(config.MODELS_DIR, f"{method}_config.txt")
    with open(config_path, "w") as f:
        f.write(f"Method: {method}\n")
        f.write("Model: RandomForestRegressor\n")
        f.write("Scaler: None\n")
        f.write(f"KDE Bandwidth: {best_params['bandwidth']:.5f}\n\n")
        f.write("Best hyperparameters from Optuna:\n")
        for key, val in best_params.items():
            f.write(f"  {key}: {val}\n")

    # === Save SHAP values ===
    if isinstance(score_types, str):
        feature_names = config.PDB_IDS
        selected_index = 3
    else:
        feature_names = ["1B41_pre-rxn_(pre)", "1B41_post-rxn_(post)"]
        for pdb_id in config.PDB_IDS[1:]:
            feature_names += [f"{pdb_id}_pre", f"{pdb_id}_post"]
        selected_index = 7

    explainer = shap.TreeExplainer(
        model,
        X_test,
        feature_names=feature_names,
        model_output="raw",
        feature_perturbation="interventional",
    )

    shap_values = explainer.shap_values(
        X_test, approximate=False, check_additivity=True
    )

    shap_df = pd.DataFrame(
        shap_values,
        columns=feature_names
    )
    shap_df = pd.concat(
        [meta_test.reset_index(drop=True), shap_df],
        axis=1
    )
    # Append raw feature value for inspection
    shap_df[f"{feature_names[selected_index]}_value"] = X_test[:, selected_index]

    shap_output_path = os.path.join(
        config.SHAP_DIR, f"{method}.csv"
    )
    shap_df.to_csv(shap_output_path, index=False)


def main():
    for score_types in [
        "non_covalent_score",
        "pre_reaction_score",
        "post_reaction_score",
        ["pre_reaction_score", "post_reaction_score"]
    ]:
        save_predictions(score_types)


if __name__ == "__main__":
    main()

