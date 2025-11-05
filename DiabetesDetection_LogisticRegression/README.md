# Diabetes Prediction Case Study

## Overview
This case study focuses on predicting diabetes in patients using Logistic Regression. The project analyzes various health indicators and medical measurements to classify patients as diabetic or non-diabetic, providing valuable insights for early diagnosis and prevention.

## Problem Statement
Predict the likelihood of diabetes in patients based on various health indicators such as glucose levels, blood pressure, BMI, and other medical measurements to enable early intervention and treatment.

## Dataset
- **File**: `diabetes.csv`
- **Features**: 
  - Pregnancies: Number of pregnancies
  - Glucose: Plasma glucose concentration
  - BloodPressure: Diastolic blood pressure (mm Hg)
  - SkinThickness: Triceps skin fold thickness (mm)
  - Insulin: 2-Hour serum insulin (mu U/ml)
  - BMI: Body mass index (weight in kg/(height in m)²)
  - DiabetesPedigreeFunction: Diabetes pedigree function
  - Age: Age in years
- **Target**: Outcome (0: No Diabetes, 1: Diabetes)
- **Size**: 768 records

## Features
- **Data Analysis**: Comprehensive exploratory data analysis with statistical summaries
- **Visualization**: 
  - Correlation heatmap for feature relationships
  - Correlation bar chart with target variable
  - Confusion matrix visualization
- **Model**: Logistic Regression with StandardScaler preprocessing
- **Evaluation**: Accuracy, Classification Report, Confusion Matrix
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Pipeline**: Scikit-learn Pipeline for streamlined workflow
- **Validation**: 80/20 train-test split with random state for reproducibility

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python Diabetespredictor.py --data diabetes.csv
```

### Command Line Arguments
- `--data`: Path to the dataset CSV file (default: diabetes.csv)

## Output
The application generates:
- **Model Performance Metrics**: Accuracy percentage and detailed classification report
- **Visualizations**: Saved in `artifacts_diabetes/plots/`
  - `correlation_heatmap.png`: Feature correlation matrix
  - `correlation_bar.png`: Feature correlation with target variable
  - `confusion_matrix.png`: Model prediction accuracy matrix
- **Trained Model**: Saved in `artifacts_diabetes/models/` with timestamp

## Model Performance
The Logistic Regression model typically achieves:
- **Accuracy**: 75-80% on test data
- **Good Precision and Recall**: For both diabetic and non-diabetic classes
- **Feature Importance**: Identifies most predictive health indicators

## Key Insights
1. **Glucose Level**: Typically the most important predictor of diabetes
2. **BMI**: Strong correlation with diabetes risk
3. **Age**: Higher age increases diabetes likelihood
4. **Medical Relevance**: Model helps in early diabetes screening

## Dataset Features Description
- **Pregnancies**: Number of times pregnant (0-17)
- **Glucose**: Plasma glucose concentration after 2 hours in an oral glucose tolerance test (0-199)
- **BloodPressure**: Diastolic blood pressure in mm Hg (0-122)
- **SkinThickness**: Triceps skinfold thickness in mm (0-99)
- **Insulin**: 2-Hour serum insulin in mu U/ml (0-846)
- **BMI**: Body mass index calculated as weight in kg/(height in m)² (0-67.1)
- **DiabetesPedigreeFunction**: Diabetes pedigree function (0.078-2.42)
- **Age**: Age in years (21-81)

## File Structure
```
Diabetes_Case_Study/
├── Diabetespredictor.py
├── diabetes.csv
├── dependencies.txt
├── requirements.txt
├── readme.txt
├── README.md
└── artifacts_diabetes/
    ├── models/
    │   └── diabetes_pipeline_*.joblib
    └── plots/
        ├── correlation_heatmap.png
        ├── correlation_bar.png
        └── confusion_matrix.png
```

## Medical Context
This model can be used for:
- **Early Screening**: Identify high-risk patients for diabetes
- **Prevention**: Help patients make lifestyle changes before diabetes develops
- **Clinical Decision Support**: Assist healthcare providers in diagnosis
- **Risk Assessment**: Evaluate diabetes risk based on multiple health factors

## Risk Factors Identified
The model typically identifies these as key risk factors:
1. **High Glucose Levels**: Primary indicator of diabetes
2. **High BMI**: Obesity increases diabetes risk
3. **Age**: Risk increases with age
4. **Blood Pressure**: Hypertension is associated with diabetes
5. **Family History**: Diabetes pedigree function indicates genetic predisposition

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- joblib >= 1.3.2

## Author
**Swaraj Santoshrao More**  
*Date: 18/09/2025*
