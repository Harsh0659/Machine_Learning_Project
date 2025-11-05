import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import argparse
from datetime import datetime


# CONFIGURATION DETAILS
ARTIFACTS = Path("artifacts_diabetes")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- load_dataset()
# Description :- Loads dataset and returns DataFrame, shape, and description
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def load_dataset(filename):
    df = pd.read_csv(filename)
    shape = df.shape
    desc = df.describe()
    return df, shape, desc


###############################################################################################################
# Function name :- correlationHeatmap()
# Description :- Generates correlation heatmap and correlation barplot
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def correlationHeatmap(df):
    corrmatrix = df.corr()

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(corrmatrix, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    # Barplot of correlation with target
    correlations = df.corr()["Outcome"].drop("Outcome")
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).index)

    plt.figure(figsize=(10, 6))
    correlations.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Feature Importance (Correlation with Outcome)")
    plt.ylabel("Correlation Coefficient")
    plt.savefig(PLOTS_DIR / "correlation_bar.png")
    plt.close()

    return corrmatrix


###############################################################################################################
# Function name :- DataSplit()
# Description :- Splits data into train and test sets
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def DataSplit(df):
    X = df.drop(columns=["Outcome"])
    y = df["Outcome"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


###############################################################################################################
# Function name :- ModelBuilding()
# Description :- Builds pipeline (StandardScaler + LogisticRegression), trains & evaluates
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def ModelBuilding(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred) * 100
    print("\nAccuracy:", accuracy)
    print("\nClassification Report:\n", classification_report(Y_test, y_pred))

    # Confusion matrix heatmap
    cm = confusion_matrix(Y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    return accuracy, pipeline


###############################################################################################################
# Function name :- PreservetheModel()
# Description :- Saves trained pipeline to disk
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"diabetes_pipeline_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved at: {model_file}")
    return model_file


###############################################################################################################
# Function name :- main()
# Description :- Entry point function for the application
# Author :- Swaraj Santoshrao More
# Date :- 18/09/2025
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Diabetes Predictor Application")
    parser.add_argument("--data", type=str, default="diabetes.csv", help="Path to dataset CSV")
    args = parser.parse_args()

    line = "*" * 84
    print(line)
    print("--------------------------------- Diabetes Predictor Application ---------------------------------")

    # Load dataset
    df, shape, desc = load_dataset(args.data)
    print("\nFirst five rows:\n", df.head())
    print("\nShape:", shape)
    print("\nStatistical Summary:\n", desc)

    # EDA
    corr = correlationHeatmap(df)
    print("\nCorrelation Matrix:\n", corr)

    # Split data
    X_train, X_test, Y_train, Y_test = DataSplit(df)

    # Model training
    acc, model = ModelBuilding(X_train, X_test, Y_train, Y_test)
    print(f"\nModel Accuracy: {acc:.2f}%")

    # Save model
    PreservetheModel(model)

    print(line)
    print("Pipeline trained & saved successfully.")
    print(f"Artifacts stored in: {ARTIFACTS.resolve()}")
    print(line)


if __name__ == "__main__":
    main()
