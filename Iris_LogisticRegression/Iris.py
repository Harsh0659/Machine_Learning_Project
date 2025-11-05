####====================================================================####
#                               Iris Case Study                           #
#                     Using Logistic Regression Classification            #
#                           Author: [Harshad Kawade]                       #
#                         Last Updated: [04-09-2025]                       #
####====================================================================####

#==========================================================================#
#                            ****Output****                                #
# Final Accuracy: …%                                                       #
# Confusion Matrix                                                         #
# …                                                                        #
# Classification Report                                                    #
# …                                                                        #
#==========================================================================#

#==========================================================================#
#                          Import Libraries                                #
#==========================================================================#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, ConfusionMatrixDisplay
)

#--------------------------------------------------------------------------#
ARTIFACTS = Path("iris_artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "logreg_pipeline.joblib"
RANDOM_STATE = 42
TEST_SIZE = 0.2
#==========================================================================#
#                      Load Dataset from CSV                               #
#==========================================================================#
def load_dataset(path):
    return pd.read_csv(path)

#==========================================================================#
#                       Clean the Data (Handle NaNs)                       #
#==========================================================================#
def clean_data(df):
    return df.dropna()

#==========================================================================#
#                      Separate Features & Target                          #
#==========================================================================#
def separate_features_target(df, target_column='species'):
    if target_column not in df.columns:
        # fallback for Kaggle / seaborn iris which uses "variety"
        if 'variety' in df.columns:
            target_column = 'variety'
        else:
            raise ValueError(f"Target column '{target_column}' not found. Columns: {list(df.columns)}")

    if df[target_column].dtype == 'object':
        le = LabelEncoder()
        df[target_column] = le.fit_transform(df[target_column])

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

#==========================================================================#
#                         Scale the Features                               #
#==========================================================================#
def scale_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)

#==========================================================================#
#                    Split into Training and Test Set                      #
#==========================================================================#
def split_data(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)

#==========================================================================#
#                 Train Logistic Regression Model                          #
#==========================================================================#
def train_logreg_model(X_train, y_train):
    model = LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs')
    model.fit(X_train, y_train)
    return model

#==========================================================================#
#                       Evaluate Model and Print Metrics                   #
#==========================================================================#
def evaluate_model(y_test, y_pred):
    print("Final Accuracy :", accuracy_score(y_test, y_pred) * 100)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report")
    print(classification_report(y_test, y_pred, digits=4))

    pre = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Precision:", pre)
    print("Recall Score :", rec)
    print("F1 Score :", f1)

#==========================================================================#
#                Plot Confusion Matrix Visualization                       #
#==========================================================================#
def plot_confusion(model, X_test, y_test):
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, cmap="Blues")
    plt.title("Confusion Matrix – Logistic Regression on Iris")
    plt.show()

#==========================================================================#
#               Train Logistic Pipeline & Save the Model                   #
#==========================================================================#
def train_logreg_pipeline_and_save(X_train, X_test, y_train, y_test, save_path=MODEL_PATH):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, multi_class='auto', solver='lbfgs'))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print("Accuracy (Logistic Regression Pipeline):", accuracy_score(y_test, y_pred) * 100)
    joblib.dump(pipe, save_path)
    print(f"Logistic Regression pipeline model saved to: {save_path}")
    plot_confusion(pipe, X_test, y_test)

#==========================================================================#
#                          Full Pipeline Call                              #
#==========================================================================#
def IrisCase(path):
    df = load_dataset(path)
    df = clean_data(df)
    X, y = separate_features_target(df)
    X_scaled = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    model = train_logreg_model(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Final Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    evaluate_model(y_test, y_pred)
    train_logreg_pipeline_and_save(X_train, X_test, y_train, y_test)

#==========================================================================#
#                               Main Call                                  #
#==========================================================================#
def main():
    # You can generate iris.csv if not present:
    # df = sns.load_dataset('iris'); df.rename(columns={'species': 'species'}, inplace=True); df.to_csv('iris.csv', index=False)
    IrisCase("iris.csv")

if __name__ == "__main__":
    main()
