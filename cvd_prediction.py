"""
AI Model for Cardiovascular Disease Prediction
================================================
Predicts cardiovascular disease (CVD) risk using three datasets:
  - Cardio Train Dataset (primary)
  - Framingham Heart Study
  - UCI Heart Disease Dataset

Author: tarajeeclarke
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────

def load_cardio_train(filepath: str) -> pd.DataFrame:
    """Load and do initial cleaning on the Cardio Train Dataset."""
    df = pd.read_csv(filepath, sep=";")
    print(f"[Cardio Train] Loaded: {df.shape[0]} rows × {df.shape[1]} cols")

    # Drop identifier column — provides no predictive value
    if "id" in df.columns:
        df.drop(columns=["id"], inplace=True)

    # Convert age from days to years for interpretability
    if df["age"].max() > 1000:
        df["age"] = (df["age"] / 365.25).round(1)

    return df


def load_framingham(filepath: str) -> pd.DataFrame:
    """Load the Framingham Heart Study dataset."""
    df = pd.read_csv(filepath)
    print(f"[Framingham]   Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


def load_uci_heart(filepath: str) -> pd.DataFrame:
    """Load the UCI Heart Disease Dataset."""
    df = pd.read_csv(filepath)
    print(f"[UCI Heart]    Loaded: {df.shape[0]} rows × {df.shape[1]} cols")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target_col: str) -> tuple:
    """
    Impute missing values, separate features / target, scale features.
    Returns: X_train, X_test, y_train, y_test, scaler
    """
    print(f"\nMissing values before imputation:\n{df.isnull().sum()}\n")

    # Impute numerical columns with median (robust to outliers)
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    print("Missing values after imputation: 0")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Standardise — zero mean, unit variance (required for logistic regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(
        f"Train set: {X_train_scaled.shape[0]} samples | "
        f"Test set: {X_test_scaled.shape[0]} samples"
    )
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# ─────────────────────────────────────────────
# 3. EDA
# ─────────────────────────────────────────────

def run_eda(df: pd.DataFrame, target_col: str, dataset_name: str = "Dataset"):
    """Generate EDA plots: class distribution + correlation heatmap."""
    print(f"\n{'='*50}")
    print(f"EDA — {dataset_name}")
    print(f"{'='*50}")
    print(df.describe())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"EDA — {dataset_name}", fontsize=14, fontweight="bold")

    # Class distribution
    sns.countplot(x=target_col, data=df, palette="Set2", ax=axes[0])
    axes[0].set_title("Target Class Distribution")
    axes[0].set_xlabel("CVD (0 = No, 1 = Yes)")
    axes[0].set_ylabel("Count")

    # Correlation heatmap
    corr = df.corr(numeric_only=True)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        linewidths=0.5,
        ax=axes[1],
    )
    axes[1].set_title("Feature Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(f"eda_{dataset_name.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()
    print(f"EDA plot saved.")


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & EVALUATION
# ─────────────────────────────────────────────

def train_and_evaluate(
    X_train, X_test, y_train, y_test,
    label: str = "Model",
    solver: str = "liblinear",
    max_iter: int = 200,
):
    """
    Train logistic regression and print full evaluation metrics.
    Returns the fitted model.
    """
    print(f"\n{'='*50}")
    print(f"Training: {label}")
    print(f"{'='*50}")

    model = LogisticRegression(
        solver=solver,
        max_iter=max_iter,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}  ({acc*100:.1f}%)")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No CVD", "CVD"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No CVD", "CVD"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(cmap="Blues", ax=ax)
    ax.set_title(f"Confusion Matrix — {label}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{label.lower().replace(' ', '_')}.png", dpi=150)
    plt.show()

    tn, fp, fn, tp = cm.ravel()
    print(f"  True Negatives : {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives : {tp}")

    return model


# ─────────────────────────────────────────────
# 5. MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Cardiovascular Disease Risk Prediction Pipeline")
    print("=" * 60)

    # ── Load primary dataset ──────────────────────────────────
    # Download from: https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
    cardio_path = "cardio_train.csv"

    try:
        df_cardio = load_cardio_train(cardio_path)
    except FileNotFoundError:
        print(
            f"\n[ERROR] '{cardio_path}' not found.\n"
            "Download it from Kaggle: "
            "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset\n"
            "and place it in this directory."
        )
        return

    # ── EDA ───────────────────────────────────────────────────
    run_eda(df_cardio, target_col="cardio", dataset_name="Cardio Train")

    # ── Preprocessing ─────────────────────────────────────────
    X_train, X_test, y_train, y_test, _ = preprocess(df_cardio, target_col="cardio")

    # ── Baseline model (default lbfgs — for comparison) ───────
    train_and_evaluate(
        X_train, X_test, y_train, y_test,
        label="Baseline lbfgs",
        solver="lbfgs",
        max_iter=100,
    )

    # ── Refined model (liblinear, 200 iterations) ─────────────
    train_and_evaluate(
        X_train, X_test, y_train, y_test,
        label="Refined liblinear",
        solver="liblinear",
        max_iter=200,
    )

    print("\nPipeline complete. Plots saved to working directory.")


if __name__ == "__main__":
    main()
