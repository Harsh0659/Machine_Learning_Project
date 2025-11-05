import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path
import joblib
from datetime import datetime
import argparse


# CONFIGURATION
ARTIFACTS = Path("artifacts_wine")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- load_dataset()
# Description :- Loads Wine dataset and performs basic preprocessing (drop nulls)
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def load_dataset(filename):
    df = pd.read_csv(filename)
    print("-" * 80)
    print("Dataset loaded successfully")
    df.dropna(inplace=True)

    print("First five rows:")
    print(df.head())
    print("-" * 80)
    print("Dataset shape:", df.shape)
    print("-" * 80)
    print("Statistical Summary:\n", df.describe())

    return df


###############################################################################################################
# Function name :- DataSplit()
# Description :- Splits dataset into train/test and scales features
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def DataSplit(df):
    X = df.drop(columns=["Class"])
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, Y_train, Y_test, scaler


###############################################################################################################
# Function name :- FindBestK()
# Description :- Finds best K value using accuracy scores
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def FindBestK(X_train, X_test, Y_train, Y_test):
    accuracy_scores = []
    k_range = range(1, 20)

    for k in k_range:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(Y_test, y_pred)
        accuracy_scores.append(accuracy)

    # Save plot Accuracy vs K
    plt.figure(figsize=(8, 5))
    plt.plot(k_range, accuracy_scores, marker="o", linestyle="--")
    plt.title("Accuracy vs K value")
    plt.xlabel("K value")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(k_range)
    plt.savefig(PLOTS_DIR / "accuracy_vs_k.png")
    plt.close()

    best_k = k_range[accuracy_scores.index(max(accuracy_scores))]
    print("Best value of K:", best_k)
    return best_k


###############################################################################################################
# Function name :- ModelBuilding()
# Description :- Builds KNN model with best K, trains & evaluates
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def ModelBuilding(X_train, X_test, Y_train, Y_test, best_k):
    model = KNeighborsClassifier(n_neighbors=best_k)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(Y_test, y_pred) * 100
    cm = confusion_matrix(Y_test, y_pred)

    print("-" * 80)
    print("Final Model Accuracy: {:.2f}%".format(acc))
    print("-" * 80)
    print("Confusion Matrix:\n", cm)
    print("-" * 80)
    print("Classification Report:\n", classification_report(Y_test, y_pred))

    return model


###############################################################################################################
# Function name :- PreservetheModel()
# Description :- Saves trained KNN model
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"wine_knn_model_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved at: {model_file}")
    return model_file


###############################################################################################################
# Function name :- main()
# Description :- Entry point for Wine Predictor using KNN
# Author :- Harshad Kawade
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Wine Predictor using KNN Classifier")
    parser.add_argument("--data", type=str, default="WinePredictor.csv", help="Path to Wine dataset CSV")
    args = parser.parse_args()

    print("-" * 80)
    print("-------------------- Wine Predictor (KNN) --------------------")

    df = load_dataset(args.data)
    X_train, X_test, Y_train, Y_test, scaler = DataSplit(df)
    best_k = FindBestK(X_train, X_test, Y_train, Y_test)
    model = ModelBuilding(X_train, X_test, Y_train, Y_test, best_k)
    PreservetheModel(model)

    print("-" * 80)
    print("Pipeline executed successfully. Artifacts stored in:", ARTIFACTS.resolve())
    print("-" * 80)


if __name__ == "__main__":
    main()
Harshad Kawade
# Date :- 18/09/2025