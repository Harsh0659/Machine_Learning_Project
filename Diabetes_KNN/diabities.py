####====================================================================####
#                          Diabetes Prediction System                      #
#                 Using K-Nearest Neighbors Classification                 #
#                           Author: [Harshad Kawade]                       #
#                         Last Updated: [03-09-2025]                       #
####====================================================================####

#==========================================================================#
#                            ****Output****                                #
# Best K Value: 18                                                         #
# Final Accuracy with Best K (18): 76.62%                                  #
# Final best Accuracy is : 76.62337662337663                               #
# Confusion Matrix                                                         #
# [[89 10]                                                                 #
#  [26 29]]                                                                #
# Classification Report                                                    #
#             precision    recall  f1-score   support                      #
#                                                                          #
#            0       0.77      0.90      0.83        99                    #
#            1       0.74      0.53      0.62        55                    #
#                                                                          #
#     accuracy                           0.77       154                    #
#    macro avg       0.76      0.71      0.72       154                    #
# weighted avg       0.76      0.77      0.76       154                    #
#                                                                          #
# ROC-AUC score: 0.7131313131313132                                        #
# Precision: 0.7435897435897436                                            #
# Recall Score : 0.5272727272727272                                        #
# F1 Score : 0.6170212765957447                                            #
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, precision_score, recall_score, f1_score
)
ARTIFACTS=Path("artifacts_sample")
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH=ARTIFACTS/"KNN_pipeline.joblib"
RANDOM_STATE=42
TEST_SIZE=0.2
#==========================================================================#
#                      Load Dataset from CSV                               #
#    Load the dataset from a CSV file.                                     #
#                                                                          #
# Args:                                                                    #
#      path (str): Path to the dataset.                                    #
#Returns:                                                                  #
#         pd.DataFrame: Loaded dataset.                                    #
#==========================================================================#
def load_dataset(path):
    return pd.read_csv(path)

#==========================================================================#
#                       Clean the Data (Handle NaNs)                       #
# Drops rows with missing values.                                          #  
#Args:                                                                     #
#     df (pd.DataFrame): Input dataframe.                                  #
#Returns:                                                                  #
#         pd.DataFrame: Cleaned dataframe.                                 #
#==========================================================================#
def clean_data(df):
    return df.dropna()

#==========================================================================#
#                      Separate Features & Target                          #
#Separates features and target column.                                     #
# Args:                                                                    #
#     df (pd.DataFrame): Cleaned dataframe.                                #
#     target_column (str): Target variable name.                           #
# Returns:                                                                 #
#          X (pd.DataFrame), y (pd.Series)                                 #
#==========================================================================#
def separate_features_target(df, target_column='Outcome'):

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return X, y

#==========================================================================#
#                         Scale the Features                               #
# Scales the features using StandardScaler.                                #
# Args:                                                                    #
#     X (pd.DataFrame): Feature matrix.                                    #
#  Returns:                                                                #
#          np.ndarray: Scaled features.                                    #
#==========================================================================#
def scale_features(X):

    scaler = StandardScaler()
    return scaler.fit_transform(X)

#==========================================================================#
#                    Split into Training and Test Set                      #
# Splits the dataset into train and test sets.                             #
#                                                                          #
#Returns:                                                                  #
#         X_train, X_test, y_train, y_test                                 #
#==========================================================================#
def split_data(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    
    return train_test_split(x, y, test_size=test_size, random_state=random_state)

#==========================================================================#
#                    Train KNN Model for a Given K                         #
# Returns:                                                                 #
#        Trained KNN model.                                                #
#==========================================================================#
def train_knn_model(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

#==========================================================================#
#                Evaluate Accuracy for Multiple K Values                   #
# Finds the best value of K for KNN based on test accuracy.                #
#                                                                          #  
# Returns:                                                                 #
#         int: Best K value.                                               #
#         list: List of accuracies for each K.                             #
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
#             Evaluates and prints various classification metrics          #
#==========================================================================#
def evaluate_model(y_test, y_pred):
   
    accuracy=accuracy_score(y_test,y_pred)
    print("Final best Accuracy is :",accuracy*100)
   
    cm=confusion_matrix(y_test,y_pred)
    print("Confusion Matrix")
    print(cm)

    CL=classification_report(y_test,y_pred)
    print("Classification Report ",CL)

    ROC_AUC = roc_auc_score(y_test, y_pred)
    print("ROC-AUC score:", ROC_AUC)

    pre= precision_score(y_test, y_pred)
    print("Precision:", pre)

    RS = recall_score(y_test, y_pred)
    print("Recall Score :",RS)

    f1 = f1_score(y_test, y_pred)
    print("F1 Score :",f1)


#==========================================================================#
#                      Plot Accuracy Curve vs. K values                    #
#==========================================================================#
def plot_accuracy_curve(accuracies, k_range):
    
    plt.figure(figsize=(10,6))
    plt.plot(k_range, accuracies, marker='o', linestyle='--', color='b')
    plt.title('KNN Accuracy vs. K Value')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.show()

#==========================================================================#
#               Train KNN Pipeline & Save the Model                        #
#                                                                          #
# Creates a scikit-learn pipeline that standardizes the features and       #
# trains a K-Nearest Neighbors classifier. Evaluates the model using       #
# accuracy and saves the trained pipeline to disk using joblib.            #
#                                                                          #
# Args:                                                                    #
#     X_train (np.ndarray or pd.DataFrame): Training features.             #
#     X_test (np.ndarray or pd.DataFrame): Testing features.               #
#     y_train (pd.Series): Training labels.                                #
#     y_test (pd.Series): Testing labels.                                  #
#     save_path (str or Path): File path to save the trained pipeline.     #
#     k (int): Number of neighbors for KNN.                                #
#==========================================================================#
def train_knn_pipeline_and_save(X_train, X_test, y_train, y_test, k, save_path=MODEL_PATH):
   
    # Define the pipeline with scaler and KNN classifier
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=k))
    ])

    # Fit the pipeline on training data
    pipe.fit(X_train, y_train)

    # Make predictions
    y_pred = pipe.predict(X_test)

    # Print evaluation metric
    print(f"Accuracy (KNN Pipeline, k={k}):", accuracy_score(y_test, y_pred) * 100)

    # Save the pipeline model to the specified path
    joblib.dump(pipe, save_path)
    print(f"KNN pipeline model saved to: {save_path}")

#==========================================================================#
#                          Full Pipeline Call                              #
#      Full pipeline to load data, train, and evaluate the KNN model.      #
#==========================================================================#
def DiabetesCase(path):
    
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
    DiabetesCase("diabetes.csv")

if __name__ == "__main__":
    main()