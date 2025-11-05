####====================================================================####
#                               Iris Case Study                           #
#                 Using K-Nearest Neighbors Classification                #
#                           Author: [Harshad Kawade]                       #
#                         Last Updated: [04-09-2025]                       #
####====================================================================####

#==========================================================================#
#                            ****Output****                                #
# Best K Value: …                                                          #
# Final Accuracy with Best K (…): …%                                       #
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_score, recall_score, f1_score
)

#--------------------------------------------------------------------------#
ARTIFACTS = Path("iris_artifacts")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "KNN_pipeline.joblib"
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
    # convert species string labels to numeric codes
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
#                    Train KNN Model for a Given K                         #
#==========================================================================#
def train_knn_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

#==========================================================================#
#                Evaluate Accuracy for Multiple K Values                   #
#==========================================================================#
def find_best_k(X_train, X_test, y_train, y_test, k_range=range(1, 25)):
    accuracy_scores = []
    for k in k_range:
        model = train_knn_model(X_train, y_train, k)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracy_scores.append(acc)
    best_k = k_range[accuracy_scores.index(max(accuracy_scores))]
    return best_k, accuracy_scores

#==========================================================================#
#                       Evaluate Model and Print Metrics                   #
#==========================================================================#
def evaluate_model(y_test, y_pred):
    print("Final best Accuracy is :", accuracy_score(y_test, y_pred) * 100)
    print("Confusion Matrix")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report")
    print(classification_report(y_test, y_pred, digits=4))

    # per-class metrics averaged (for multi-class just macro average)
    pre = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Precision:", pre)
    print("Recall Score :", rec)
    print("F1 Score :", f1)

#==========================================================================#
#                      Plot Accuracy Curve vs. K values                    #
#==========================================================================#
def plot_accuracy_curve(accuracies, k_range):
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, accuracies, marker='o', linestyle='--', color='b')
    plt.title('KNN Accuracy vs. K Value (Iris)')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

#==========================================================================#
#               Train KNN Pipeline & Save the Model                        #
#==========================================================================#
def train_knn_pipeline_and_save(X_train, X_test, y_train, y_test, k, save_path=MODEL_PATH):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"Accuracy (KNN Pipeline, k={k}):", accuracy_score(y_test, y_pred) * 100)
    joblib.dump(pipe, save_path)
    print(f"KNN pipeline model saved to: {save_path}")

#==========================================================================#
#                          Full Pipeline Call                              #
#==========================================================================#
def IrisCase(path):
    df = load_dataset(path)
    df = clean_data(df)
    X, y = separate_features_target(df)
    X_scaled = scale_features(X)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    best_k, accuracies = find_best_k(X_train, X_test, y_train, y_test)
    print(f"Best K Value: {best_k}")

    train_knn_pipeline_and_save(X_train, X_test, y_train, y_test, k=best_k)

    model = train_knn_model(X_train, y_train, best_k)
    y_pred = model.predict(X_test)

    print(f"Final Accuracy with Best K ({best_k}): {accuracy_score(y_test, y_pred) * 100:.2f}%")
    evaluate_model(y_test, y_pred)
    plot_accuracy_curve(accuracies, range(1, 25))

#==========================================================================#
#                               Main Call                                  #
#==========================================================================#
def main():
    # Replace with the correct csv path or use seaborn’s iris if no CSV is available:
    # df = sns.load_dataset('iris'); df.to_csv('iris.csv', index=False)
    IrisCase("iris.csv")

if __name__ == "__main__":
    main()
