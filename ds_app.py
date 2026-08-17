from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from train import TARGET, TitanicClassifier, engineer_features


st.set_page_config(
    page_title="Titanic Survival Classifier",
    page_icon="🚢",
    layout="wide",
)


@st.cache_resource
def load_artifacts(
    model_path: str,
    preprocessor_path: str,
    statistics_path: str,
    model_modified: float,
    preprocessor_modified: float,
    statistics_modified: float,
):
    del model_modified, preprocessor_modified, statistics_modified

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
    preprocessor = joblib.load(preprocessor_path)

    with Path(statistics_path).open(encoding="utf-8") as file:
        statistics = json.load(file)

    model = TitanicClassifier(
        input_dim=int(checkpoint["input_dim"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, preprocessor, statistics, checkpoint, device


def load_csv(uploaded_file, csv_path: str) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)

    if not csv_path.strip():
        raise ValueError("Upload a CSV or provide a CSV path.")

    path = Path(csv_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    return pd.read_csv(path)


def predict(
    raw_df: pd.DataFrame,
    model: TitanicClassifier,
    preprocessor,
    statistics: dict,
    device: torch.device,
) -> np.ndarray:
    features = engineer_features(raw_df, statistics)
    transformed = preprocessor.transform(features).astype(np.float32)

    with torch.no_grad():
        tensor = torch.as_tensor(transformed, dtype=torch.float32, device=device)
        return torch.sigmoid(model(tensor)).cpu().numpy()


def metric_values(y_true: np.ndarray, y_pred: np.ndarray, probabilities: np.ndarray):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )

    values = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }

    if len(np.unique(y_true)) == 2:
        values["ROC-AUC"] = roc_auc_score(y_true, probabilities)

    return values


st.title("Titanic Survival Classification")
st.write(
    "Load the artifacts created by `train.py`, provide a raw Titanic-format CSV, "
    "and run PyTorch inference. If the CSV contains `Survived`, the app also "
    "shows evaluation metrics."
)

with st.sidebar:
    st.header("Model artifacts")
    artifact_dir = st.text_input("Artifact directory", "artifacts")
    model_path = st.text_input(
        "Model path",
        str(Path(artifact_dir) / "titanic_model.pt"),
    )
    preprocessor_path = st.text_input(
        "Preprocessor path",
        str(Path(artifact_dir) / "preprocessor.joblib"),
    )
    statistics_path = st.text_input(
        "Feature statistics path",
        str(Path(artifact_dir) / "feature_statistics.json"),
    )
    st.caption("Load only model and preprocessing files that you trust.")

st.header("Input data")
source_column, path_column = st.columns(2)

with source_column:
    uploaded_file = st.file_uploader(
        "Upload a raw Titanic CSV",
        type=["csv"],
    )

with path_column:
    csv_path = st.text_input(
        "Or provide a server-side CSV path",
        str(Path("artifacts") / "validation_data.csv"),
    )

threshold = st.slider(
    "Classification threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
)

if st.button("Run inference", type="primary"):
    try:
        artifact_paths = [
            Path(model_path).expanduser(),
            Path(preprocessor_path).expanduser(),
            Path(statistics_path).expanduser(),
        ]
        missing_artifacts = [str(path) for path in artifact_paths if not path.is_file()]
        if missing_artifacts:
            raise FileNotFoundError(
                "Missing model artifact(s): " + ", ".join(missing_artifacts)
            )

        model, preprocessor, statistics, checkpoint, device = load_artifacts(
            str(artifact_paths[0]),
            str(artifact_paths[1]),
            str(artifact_paths[2]),
            *(path.stat().st_mtime for path in artifact_paths),
        )
        raw_df = load_csv(uploaded_file, csv_path)
        probabilities = predict(
            raw_df,
            model,
            preprocessor,
            statistics,
            device,
        )
        predictions = (probabilities >= threshold).astype(int)

        result_columns = [
            column for column in ["PassengerId", "Name", TARGET]
            if column in raw_df.columns
        ]
        results = raw_df[result_columns].copy()
        results["SurvivalProbability"] = probabilities
        results["Prediction"] = predictions

        st.success(f"Inference completed for {len(results)} passengers.")

        config = checkpoint.get("config", {})
        st.caption(
            f"Model: {config.get('name', 'PyTorch classifier')} · "
            f"Device: {device} · Threshold: {threshold:.2f}"
        )

        if TARGET in raw_df.columns:
            y_true = raw_df[TARGET].to_numpy(dtype=int)
            metrics = metric_values(y_true, predictions, probabilities)
            metric_columns = st.columns(len(metrics))

            for column, (name, value) in zip(metric_columns, metrics.items()):
                column.metric(name, f"{value:.3f}")

            plot_column, roc_column = st.columns(2)

            with plot_column:
                matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
                figure, axis = plt.subplots(figsize=(5, 4))
                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Did not survive", "Survived"],
                    yticklabels=["Did not survive", "Survived"],
                    ax=axis,
                )
                axis.set_title("Confusion Matrix")
                axis.set_xlabel("Predicted")
                axis.set_ylabel("Actual")
                st.pyplot(figure)
                plt.close(figure)

            with roc_column:
                if len(np.unique(y_true)) == 2:
                    false_positive_rate, true_positive_rate, _ = roc_curve(
                        y_true,
                        probabilities,
                    )
                    figure, axis = plt.subplots(figsize=(5, 4))
                    axis.plot(
                        false_positive_rate,
                        true_positive_rate,
                        label=f"AUC = {roc_auc_score(y_true, probabilities):.3f}",
                    )
                    axis.plot([0, 1], [0, 1], "--", color="gray")
                    axis.set_title("ROC Curve")
                    axis.set_xlabel("False Positive Rate")
                    axis.set_ylabel("True Positive Rate")
                    axis.legend()
                    st.pyplot(figure)
                    plt.close(figure)
                else:
                    st.info("ROC-AUC requires both target classes.")

        st.subheader("Predictions")
        st.dataframe(results, use_container_width=True)
        st.download_button(
            "Download predictions",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="titanic_predictions.csv",
            mime="text/csv",
        )

    except Exception as error:
        st.error(str(error))
        st.exception(error)
