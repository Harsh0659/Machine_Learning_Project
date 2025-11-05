import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import joblib
from datetime import datetime
import argparse


# CONFIGURATION
ARTIFACTS = Path("artifacts_telco")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- load_dataset()
# Description :- Loads the Telco Churn dataset and performs preprocessing
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def load_dataset(filename):
    df = pd.read_csv(filename)

    # Drop irrelevant columns
    df.drop(["customerID", "gender"], axis=1, inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include="object"):
        df[col] = LabelEncoder().fit_transform(df[col])

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
# Function name :- DataSplit()
# Description :- Splits the dataset into train and test sets
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def DataSplit(df):
    X = df.drop("Churn", axis=1)
    Y = df["Churn"]

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)

    return X_train, X_test, Y_train, Y_test


###############################################################################################################
# Function name :- ModelBuilding()
# Description :- Builds Random Forest pipeline, trains and evaluates the model
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def ModelBuilding(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42))
    ])

    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred) * 100
    print("-" * 80)
    print("Model Accuracy: {:.2f}%".format(accuracy))
    print("-" * 80)
    print("Classification Report:\n", classification_report(Y_test, y_pred))

    return pipeline, y_pred


###############################################################################################################
# Function name :- featureImportance()
# Description :- Plots and saves feature importance from the Random Forest model
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def featureImportance(model, X):
    rf_model = model.named_steps["rf"]

    importance = pd.Series(rf_model.feature_importances_, index=X.columns)
    importance = importance.sort_values(ascending=False)

    plt.figure(figsize=(12, 6))
    importance.plot(kind="bar", title="Feature Importance (Random Forest)", color="skyblue", edgecolor="black")
    plt.tight_layout()
    plot_path = PLOTS_DIR / "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Feature importance plot saved at: {plot_path}")
    return importance


###############################################################################################################
# Function name :- PreservetheModel()
# Description :- Saves trained pipeline to disk
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"telco_randomforest_pipeline_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"Model saved at: {model_file}")
    return model_file


###############################################################################################################
# Function name :- main()
# Description :- Entry point for Telco Customer Churn Prediction Case Study
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Telco Customer Churn Prediction using Random Forest")
    parser.add_argument("--data", type=str, default="Telco-Customer-Churn.csv", help="Path to Telco Churn dataset")
    args = parser.parse_args()

    print("-" * 80)
    print("-------------------- Telco Customer Churn Prediction --------------------")

    # Load dataset
    df = load_dataset(args.data)

    # Split data
    X_train, X_test, Y_train, Y_test = DataSplit(df)

    # Train model
    model, y_pred = ModelBuilding(X_train, X_test, Y_train, Y_test)

    # Feature importance
    featureImportance(model, X_train)

    # Save model
    PreservetheModel(model)

    print("-" * 80)
    print("Pipeline executed successfully. Artifacts stored in:", ARTIFACTS.resolve())
    print("-" * 80)


if __name__ == "__main__":
    main()
