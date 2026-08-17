from __future__ import annotations

# ============================== Imports ==============================

import argparse
import json
import os
import random
from itertools import product
from pathlib import Path
from typing import Any

import joblib
import kagglehub
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm


# ============================== Features ==============================

TARGET = "Survived"
CATEGORICAL_COLUMNS = ["Sex"]
BASE_NUMERIC_COLUMNS = [
    "Pclass", "Age", "IsMarried", "IsChildTitle", "TicketGroupSize",
    "PossibleRelativeOnBoard", "SpouseOnBoard", "AgeCategory",
]

SELECTED_INTERACTIONS = [
    ("IsChildTitle", "PossibleRelativeOnBoard"), ("SibSp", "IsChildTitle"),
    ("IsChildTitle", "TicketGroupSize"), ("Fare", "IsChildTitle"),
    ("SibSp", "SpouseOnBoard"), ("SibSp", "IsMarried"),
    ("Pclass", "Parch"), ("Parch", "SpouseOnBoard"),
    ("PossibleRelativeOnBoard", "MaleAge18To34"), ("Parch", "MaleAge18To34"),
    ("IsChildTitle", "SpouseOnBoard"), ("IsMarried", "IsChildTitle"),
    ("Parch", "IsChildTitle"), ("TicketGroupSize", "SpouseOnBoard"),
    ("Age", "IsChildTitle"), ("Pclass", "SpouseOnBoard"),
    ("SibSp", "Parch"), ("Pclass", "AgeCategory"),
    ("Pclass", "IsChildTitle"), ("Age", "SpouseOnBoard"),
    ("SpouseOnBoard", "AgeCategory"), ("SpouseOnBoard", "MaleAge18To34"),
    ("Parch", "PossibleRelativeOnBoard"), ("IsMarried", "MaleAge18To34"),
    ("PossibleRelativeOnBoard", "SpouseOnBoard"), ("IsMarried", "SpouseOnBoard"),
    ("Pclass", "TicketGroupSize"), ("IsChildTitle", "MaleAge18To34"),
    ("IsChildTitle", "AgeCategory"), ("AgeCategory", "MaleAge18To34"),
    ("TicketGroupSize", "MaleAge18To34"), ("IsMarried", "PossibleRelativeOnBoard"),
    ("Fare", "SpouseOnBoard"), ("Parch", "IsMarried"),
    ("Pclass", "MaleAge18To34"),
]

INTERACTION_COLUMNS = [f"{first}_x_{second}" for first, second in SELECTED_INTERACTIONS]
NUMERIC_COLUMNS = BASE_NUMERIC_COLUMNS + INTERACTION_COLUMNS


# ============================== Search space ==============================

CONFIGS = [
    ("linear", (), 0.0, 1e-2, 1e-4, 64),
    ("mlp_16", (16,), 0.10, 3e-3, 1e-4, 64),
    ("mlp_32_regularized", (32,), 0.30, 1e-3, 1e-3, 32),
    ("mlp_64", (64,), 0.30, 1e-3, 1e-3, 32),
    ("mlp_64_32", (64, 32), 0.30, 1e-3, 1e-3, 32),
    ("mlp_64_32_small_batch", (64, 32), 0.35, 5e-4, 1e-3, 16),
    ("mlp_128_64", (128, 64), 0.40, 5e-4, 1e-3, 16),
    ("mlp_64_32_16", (64, 32, 16), 0.40, 5e-4, 1e-3, 16),
    ("mlp_128_64_32", (128, 64, 32), 0.45, 3e-4, 1e-3, 16),
]


SEARCH_SPACE = [
    dict(zip(
        ["name", "hidden_dims", "dropout", "learning_rate", "weight_decay", "batch_size"],
        config
    ))
    for config in CONFIGS
]
MAX_EPOCHS = 500
PATIENCE = 30


# ============================== Model ==============================

class TitanicClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        for output_dim in hidden_dims:
            layers += [nn.Linear(input_dim, output_dim), nn.ReLU(), nn.BatchNorm1d(output_dim), nn.Dropout(dropout)]
            input_dim = output_dim

        self.network = nn.Sequential(*layers, nn.Linear(input_dim, 1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


# ============================== Utilities ==============================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def age_statistics(df: pd.DataFrame) -> dict[str, Any]:
    titles = (
        df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
        .replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    )
    medians = pd.DataFrame({"Title": titles, "Pclass": df["Pclass"], "Age": df["Age"]}) \
        .groupby(["Title", "Pclass"])["Age"].median().dropna()

    return {
        "overall_age_median": float(df["Age"].median()),
        "age_medians": {f"{title}|{int(pclass)}": float(age) for (title, pclass), age in medians.items()},
    }


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
        ("numeric", StandardScaler(), NUMERIC_COLUMNS),
    ])


def engineer_features(raw_df: pd.DataFrame, statistics: dict[str, Any]) -> pd.DataFrame:
    required = {"Name", "Sex", "Pclass", "Age", "SibSp", "Parch", "Ticket", "Fare"}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}")

    df = raw_df.copy()
    df["_RowId"] = df.get("PassengerId", pd.Series(np.arange(len(df)), index=df.index))
    df["Title"] = df["Name"].str.extract(r",\s*([^.]*)\.", expand=False).str.strip()
    df["Surname"] = df["Name"].str.split(",").str[0].str.strip()
    df["AgeTitle"] = df["Title"].replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})

    age_keys = df["AgeTitle"] + "|" + df["Pclass"].astype(str)
    df["Age"] = df["Age"].fillna(age_keys.map(statistics["age_medians"])).fillna(statistics["overall_age_median"])
    df["Fare"] = np.log1p(df["Fare"].clip(lower=0))
    df["IsMarried"] = df["Title"].isin(["Mrs", "Mme"]).astype(int)
    df["IsChildTitle"] = df["Title"].eq("Master").astype(int)
    df["TicketGroupSize"] = df.groupby("Ticket")["_RowId"].transform("count")
    df["PossibleRelativeOnBoard"] = df.groupby(["Ticket", "Surname"])["_RowId"].transform("count").gt(1).astype(int)

    df["SpouseOnBoard"] = 0
    name_to_id = df.set_index("Name")["_RowId"].to_dict()
    wives = df[df["Title"].isin(["Mrs", "Mme"])].copy()
    wives["ExpectedHusbandName"] = (
        wives["Name"].str.replace(r"\s*\([^()]*\)", "", regex=True)
        .str.replace(r",\s*(Mrs|Mme)\.", ", Mr.", regex=True).str.strip()
    )

    for _, wife in wives.iterrows():
        husband_id = name_to_id.get(wife["ExpectedHusbandName"])
        if husband_id is None:
            candidates = df[
                df["Sex"].eq("male") & df["Surname"].eq(wife["Surname"])
                & df["Ticket"].eq(wife["Ticket"]) & df["SibSp"].gt(0)
            ]
            if len(candidates) == 1 and wife["SibSp"] > 0:
                husband_id = candidates.iloc[0]["_RowId"]

        if husband_id is not None:
            wife_mask, husband_mask = df["_RowId"].eq(wife["_RowId"]), df["_RowId"].eq(husband_id)
            df.loc[wife_mask | husband_mask, "SpouseOnBoard"] = 1
            df.loc[husband_mask, "IsMarried"] = 1

    df["AgeCategory"] = pd.cut(
        df["Age"], [0, 12, 17, 34, 59, float("inf")], labels=[0, 1, 2, 3, 4], include_lowest=True
    ).astype(int)
    df["MaleAge18To34"] = (
        df["Sex"].eq("male") & df["Age"].between(18, 34)
        & (df["Parch"].gt(0) | df["SibSp"].gt(0))
    ).astype(int)

    for first, second in SELECTED_INTERACTIONS:
        df[f"{first}_x_{second}"] = df[first] * df[second]

    return df[CATEGORICAL_COLUMNS + NUMERIC_COLUMNS]


def train_model(
    X_train: np.ndarray, y_train: np.ndarray, config: dict[str, Any], device: torch.device,
    epochs: int, seed: int, X_stop: np.ndarray | None = None, y_stop: np.ndarray | None = None,
) -> tuple[TitanicClassifier, int]:
    set_seed(seed)
    model = TitanicClassifier(X_train.shape[1], config["hidden_dims"], config["dropout"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    positive, negative = max(float(y_train.sum()), 1.0), max(float(len(y_train) - y_train.sum()), 1.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=config['batch_size'], shuffle=True
    )

    best_loss, best_state, best_epoch, stale = float("inf"), None, epochs, 0
    stop_X = torch.tensor(X_stop, device=device) if X_stop is not None else None
    stop_y = torch.tensor(y_stop, device=device) if y_stop is not None else None

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()

        if stop_X is None:
            continue

        model.eval()
        with torch.no_grad():
            stop_loss = criterion(model(stop_X), stop_y).item()

        if stop_loss < best_loss - 1e-4:
            best_loss, best_epoch, stale = stop_loss, epoch, 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def predict(model: TitanicClassifier, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X, device=device))).cpu().numpy()


def metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    predictions = (probabilities >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, predictions, average="binary", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, predictions)), "precision": float(precision),
        "recall": float(recall), "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(y_true, predictions).tolist(),
    }


# ============================== Main workflow ==============================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train and tune a PyTorch model on Kaggle Titanic train.csv.")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    load_dotenv(args.env_file)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.csv"
    if not train_path.exists():
        if not os.getenv("KAGGLE_API_TOKEN"):
            raise RuntimeError("KAGGLE_API_TOKEN was not found in the selected .env file.")
        result = Path(kagglehub.competition_download("titanic", path="train.csv", output_dir=str(args.data_dir)))
        train_path = result / "train.csv" if result.is_dir() else result

    full_df = pd.read_csv(train_path)
    train_df, validation_df = train_test_split(
        full_df, test_size=args.validation_size, random_state=args.seed, stratify=full_df[TARGET]
    )
    train_df, validation_df = train_df.reset_index(drop=True), validation_df.reset_index(drop=True)
    validation_df.to_csv(args.output_dir / "validation_data.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splitter = StratifiedKFold(args.cv_folds, shuffle=True, random_state=args.seed)
    tuning_results = []

    search = tqdm(SEARCH_SPACE, desc="Hyperparameter search", unit="config")
    for config in search:
        fold_scores, fold_epochs = [], []

        for fold, (fit_idx, stop_idx) in enumerate(splitter.split(train_df, train_df[TARGET]), 1):
            fit_df, stop_df = train_df.iloc[fit_idx], train_df.iloc[stop_idx]
            statistics = age_statistics(fit_df)
            fit_features, stop_features = engineer_features(fit_df, statistics), engineer_features(stop_df, statistics)
            preprocessor = build_preprocessor()
            X_fit = preprocessor.fit_transform(fit_features).astype(np.float32)
            X_stop = preprocessor.transform(stop_features).astype(np.float32)
            y_fit = fit_df[TARGET].to_numpy(np.float32)
            y_stop = stop_df[TARGET].to_numpy(np.float32)

            model, best_epoch = train_model(
                X_fit, y_fit, config, device, MAX_EPOCHS, args.seed + fold, X_stop, y_stop
            )
            fold_scores.append(metrics(y_stop, predict(model, X_stop, device)))
            fold_epochs.append(best_epoch)

        result = {
            "config": config,
            "mean_f1": float(np.mean([score["f1"] for score in fold_scores])),
            "mean_accuracy": float(np.mean([score["accuracy"] for score in fold_scores])),
            "mean_roc_auc": float(np.mean([score["roc_auc"] for score in fold_scores])),
            "recommended_epochs": max(1, round(np.mean(fold_epochs))),
        }
        tuning_results.append(result)
        search.set_postfix(f1=f"{result['mean_f1']:.3f}", best=f"{max(x['mean_f1'] for x in tuning_results):.3f}")

    tuning_results.sort(key=lambda item: (item["mean_f1"], item["mean_roc_auc"]), reverse=True)
    best = tuning_results[0]
    statistics = age_statistics(train_df)
    train_features = engineer_features(train_df, statistics)
    validation_features = engineer_features(validation_df, statistics)
    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_features).astype(np.float32)
    X_validation = preprocessor.transform(validation_features).astype(np.float32)
    y_train = train_df[TARGET].to_numpy(np.float32)
    y_validation = validation_df[TARGET].to_numpy(np.float32)

    model, _ = train_model(
        X_train, y_train, best["config"], device, best["recommended_epochs"], args.seed
    )
    probabilities = predict(model, X_validation, device)
    validation_metrics = metrics(y_validation, probabilities)

    checkpoint = {
        "model_state_dict": model.state_dict(), "input_dim": X_train.shape[1],
        "hidden_dims": best["config"]["hidden_dims"], "dropout": best["config"]["dropout"],
        "threshold": 0.5, "feature_names": preprocessor.get_feature_names_out().tolist(),
        "config": best["config"], "seed": args.seed,
    }
    torch.save(checkpoint, args.output_dir / "titanic_model.pt")
    joblib.dump(preprocessor, args.output_dir / "preprocessor.joblib")
    (args.output_dir / "feature_statistics.json").write_text(json.dumps(statistics, indent=2), encoding="utf-8")

    report = {
        "train_rows": len(train_df), "validation_rows": len(validation_df), "device": str(device),
        "best_config": best["config"], "recommended_epochs": best["recommended_epochs"],
        "tuning_results": tuning_results, "validation_metrics": validation_metrics,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    predictions = validation_df[["PassengerId", TARGET]].copy()
    predictions["SurvivalProbability"] = probabilities
    predictions["Prediction"] = (probabilities >= 0.5).astype(int)
    predictions.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    print(json.dumps({"best_config": best["config"], "validation_metrics": validation_metrics}, indent=2))
    print(f"Saved artifacts to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
