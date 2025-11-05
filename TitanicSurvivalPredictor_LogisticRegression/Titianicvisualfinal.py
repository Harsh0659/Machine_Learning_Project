import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
import joblib
from datetime import datetime
import argparse


# CONFIGURATION
ARTIFACTS = Path("artifacts_titanic")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- load_dataset()
# Description :- Loads Titanic dataset and preprocesses (drop cols, encode categorical, handle nulls)
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def load_dataset(filename):
    df = pd.read_csv(filename)

    # Drop unused columns
    drop_cols = [col for col in ["PassengerId", "zero"] if col in df.columns]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Handle categorical
    for col in df.select_dtypes(include="object"):
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # Fill nulls
    df.fillna(df.median(numeric_only=True), inplace=True)

    print("-" * 80)
    print("First five rows of dataset:")
    print(df.head())
    print("-" * 80)
    print("Dataset shape:", df.shape)
    print("-" * 80)
    print("Statistical Summary:")
    print(df.describe())

    return df


###############################################################################################################
# Function name :- create_visualizations()
# Description :- Generates and saves visual plots for Titanic dataset
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def create_visualizations(df):
    # Survival distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Survived", data=df)
    plt.title("Survival Distribution")
    plt.savefig(PLOTS_DIR / "survival_distribution.png")
    plt.close()

    # By Gender
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Survived", hue="Sex", data=df)
    plt.title("Survival by Gender")
    plt.savefig(PLOTS_DIR / "survival_by_gender.png")
    plt.close()

    # By Class
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Survived", hue="Pclass", data=df)
    plt.title("Survival by Passenger Class")
    plt.savefig(PLOTS_DIR / "survival_by_class.png")
    plt.close()

    # Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    print(f"Plots saved at: {PLOTS_DIR}")


###############################################################################################################
# Function name :- DataSplit()
# Description :- Splits dataset into train/test
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def DataSplit(df):
    X = df.drop("Survived", axis=1)
    y = df["Survived"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, Y_train, Y_test


###############################################################################################################
# Function name :- ModelBuilding()
# Description :- Builds Logistic Regression pipeline, trains & evaluates
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def ModelBuilding(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(Y_test, y_pred) * 100
    cm = confusion_matrix(Y_test, y_pred)

    print("-" * 80)
    print("Model Accuracy: {:.2f}%".format(acc))
    print("-" * 80)
    print("Confusion Matrix:\n", cm)
    print("-" * 80)
    print("Classification Report:\n", classification_report(Y_test, y_pred))

    return pipeline


###############################################################################################################
# Function name :- PreservetheModel()
# Description :- Saves trained pipeline
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"titanic_pipeline_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved at: {model_file}")
    return model_file


###############################################################################################################
# Function name :- main()
# Description :- Entry point for Titanic Survival Analysis
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Titanic Survival Analysis and Prediction")
    parser.add_argument("--data", type=str, default="MarvellousTitanicDataset.csv", help="Path to Titanic dataset CSV")
    args = parser.parse_args()

    print("-" * 80)
    print("-------------------- Titanic Survival Prediction --------------------")

    df = load_dataset(args.data)
    create_visualizations(df)
    X_train, X_test, Y_train, Y_test = DataSplit(df)
    model = ModelBuilding(X_train, X_test, Y_train, Y_test)
    PreservetheModel(model)

    print("-" * 80)
    print("Pipeline executed successfully. Artifacts stored in:", ARTIFACTS.resolve())
    print("-" * 80)


if __name__ == "__main__":
    main()
