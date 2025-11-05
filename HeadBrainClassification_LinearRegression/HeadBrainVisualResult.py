import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
import joblib
import argparse
from datetime import datetime


# CONFIGURATION DETAILS
ARTIFACTS = Path("artifacts_head_brain")
PLOTS_DIR = ARTIFACTS / "plots"
MODELS_DIR = ARTIFACTS / "models"
ARTIFACTS.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


###############################################################################################################
# Function name :- load_dataset()
# Description :- Loads dataset and prints sample + summary
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def load_dataset(filename):
    df = pd.read_csv(filename)
    print("-" * 84)
    print("First few records of dataset are")
    print(df.head())
    print("-" * 84)
    print("Statistical summary:\n", df.describe())
    return df


###############################################################################################################
# Function name :- DataSplit()
# Description :- Splits the dataset into train and test sets
# Author :-Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def DataSplit(df):
    X = df[["Head Size(cm^3)"]]
    y = df[["Brain Weight(grams)"]]

    print("Independent variable: Head Size")
    print("Dependent variable: Brain Weight")
    print("Total records in dataset:", X.shape)

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Dimension of training dataset:", X_train.shape)
    print("Dimension of testing dataset:", X_test.shape)

    return X_train, X_test, Y_train, Y_test


###############################################################################################################
# Function name :- ModelBuilding()
# Description :- Builds pipeline (StandardScaler + LinearRegression), trains & evaluates
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def ModelBuilding(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])

    pipeline.fit(X_train, Y_train)
    y_pred = pipeline.predict(X_test)

    mse = mean_squared_error(Y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(Y_test, y_pred)

    print("\nResult of case study")
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R² value:", r2)

    # Visual regression plot
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, Y_test, color="blue", label="Actual")
    plt.plot(X_test.values.flatten(), y_pred, color="red", linewidth=2, label="Regression Line")
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weight(grams)")
    plt.title("Head Size vs Brain Weight")
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOTS_DIR / "regression_line.png")
    plt.close()

    return pipeline, y_pred, (mse, rmse, r2)


###############################################################################################################
# Function name :- featureImportance()
# Description :- Extracts slope & intercept from Linear Regression model
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def featureImportance(model):
    regressor = model.named_steps["regressor"]
    slope = regressor.coef_[0]
    intercept = regressor.intercept_

    print("Slope of line (m):", slope)
    print("Intercept (c):", intercept)

    return slope, intercept


###############################################################################################################
# Function name :- PreservetheModel()
# Description :- Saves trained pipeline to disk
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def PreservetheModel(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = MODELS_DIR / f"head_brain_pipeline_{timestamp}.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved at: {model_file}")
    return model_file


###############################################################################################################
# Function name :- main()
# Description :- Entry point for Head–Brain Linear Regression Case Study
# Author :- Harshad Kawade
# Date :- 18/09/2025
###############################################################################################################
def main():
    parser = argparse.ArgumentParser(description="Head–Brain Linear Regression Case Study")
    parser.add_argument("--data", type=str, default="MarvellousHeadBrain.csv", help="Path to dataset CSV")
    args = parser.parse_args()

    line = "-" * 84
    print(line)
    print("--------------------------- Head–Brain Linear Regression Application ---------------------------")

    # Load dataset
    df = load_dataset(args.data)

    # Train/test split
    X_train, X_test, Y_train, Y_test = DataSplit(df)

    # Model building
    model, y_pred, metrics_values = ModelBuilding(X_train, X_test, Y_train, Y_test)

    # Feature importance
    featureImportance(model)

    # Save model
    PreservetheModel(model)

    print(line)
    print("Pipeline trained & saved successfully.")
    print(f"Artifacts stored in: {ARTIFACTS.resolve()}")
    print(line)


if __name__ == "__main__":
    main()
