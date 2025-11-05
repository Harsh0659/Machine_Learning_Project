import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import argparse
from datetime import datetime


# CONFIGURATION DETAILS
ARTIFACTS = Path("artifacts_breast_cancer")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- load_dataset()
# Description :- Loads the breast cancer dataset and returns DataFrame with target column
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def load_dataset():
    dataset = load_breast_cancer()
    X = dataset.data
    Y = dataset.target

    df = pd.DataFrame(X, columns=dataset.feature_names)
    df["target"] = Y

    print("Dataset Shape:", df.shape)
    print("Classes:", dict(zip(dataset.target_names, [0, 1])))
    print("\nFirst five rows:\n", df.head())
    return df, dataset.feature_names, dataset.target_names


###############################################################################################################
# Function name :- correlationHeatmap()
# Description :- Generates correlation heatmap for features + target
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def correlationHeatmap(df):
    corrmatrix = df.corr()

    plt.figure(figsize=(12, 8))
    sns.heatmap(corrmatrix, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig(PLOTS_DIR / "correlation_heatmap.png")
    plt.close()

    return corrmatrix


###############################################################################################################
# Function name :- DataSplit()
# Description :- Splits the dataset into train and test sets
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def DataSplit(df):
    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=0.2, random_state=42)


###############################################################################################################
# Function name :- ModelBuilding()
# Description :- Builds pipeline (StandardScaler + Linear SVC), trains & evaluates
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def ModelBuilding(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="linear", C=1))
    ])

    pipeline.fit(X_train, Y_train)
    Y_pred = pipeline.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred) * 100
    print("\nAccuracy:", acc)
    print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

    # Confusion matrix heatmap
    cm = confusion_matrix(Y_test, Y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["malignant", "benign"], yticklabels=["malignant", "benign"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(PLOTS_DIR / "confusion_matrix.png")
    plt.close()

    return acc, pipeline


###############################################################################################################
# Function name :- featureImportance()
# Description :- Plots feature importance from linear SVM coefficients
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def featureImportance(model, feature_names):
    clf = model.named_steps["classifier"]
    coefficients = clf.coef_[0]
    importance = pd.Series(coefficients, index=feature_names).sort_values(key=abs, ascending=False)

    plt.figure(figsize=(12, 6))
    importance.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Feature Importance (SVM Coefficients)")
    plt.ylabel("Coefficient Value")
    plt.savefig(PLOTS_DIR / "feature_importance.png")
    plt.close()

    print("\nModel-based Feature Importance:\n", importance.head(10))
    return importance


###############################################################################################################
# Function name :- PreservetheModel()
# Description :- Saves the trained pipeline to disk
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"breast_cancer_pipeline_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved at: {model_file}")
    return model_file


###############################################################################################################
# Function name :- main()
# Description :- Entry point of the application
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def main():
    line = "*" * 84
    print(line)
    print("--------------------------- Breast Cancer Predictor Application ---------------------------")

    # Load dataset
    df, feature_names, target_names = load_dataset()

    # Correlation heatmap
    correlationHeatmap(df)

    # Split data
    X_train, X_test, Y_train, Y_test = DataSplit(df)

    # Model training & evaluation
    acc, model = ModelBuilding(X_train, X_test, Y_train, Y_test)

    # Feature importance
    featureImportance(model, feature_names)

    # Save model
    PreservetheModel(model)

    print(line)
    print("Pipeline trained & saved successfully.")
    print(f"Artifacts stored in: {ARTIFACTS.resolve()}")
    print(line)


if __name__ == "__main__":
    main()
