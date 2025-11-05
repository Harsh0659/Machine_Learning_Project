# Breast Cancer Classification Case Study

## Overview
This case study focuses on binary classification of breast cancer tumors as malignant or benign using Support Vector Machine (SVM). The project uses the famous Wisconsin Breast Cancer dataset and provides comprehensive analysis through data visualization and model evaluation.

## Problem Statement
Classify breast cancer tumors as malignant (cancerous) or benign (non-cancerous) based on various cell nucleus measurements to assist in early diagnosis and treatment planning.

## Dataset
- **Source**: Scikit-learn's built-in Wisconsin Breast Cancer dataset
- **Features**: 30 numerical features describing cell nucleus characteristics
  - Radius, Texture, Perimeter, Area, Smoothness, Compactness, Concavity, etc.
  - Each feature has mean, standard error, and worst (largest) values
- **Target**: Binary classification (0: Malignant, 1: Benign)
- **Size**: 569 samples

## Features
- **Data Analysis**: Comprehensive exploratory data analysis with statistical summaries
- **Visualization**: 
  - Correlation heatmap for feature relationships
  - Confusion matrix visualization
  - Feature importance analysis
- **Model**: Linear SVM with StandardScaler preprocessing
- **Evaluation**: Accuracy, Classification Report, Confusion Matrix
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: Support Vector Machine (Linear Kernel)
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
python BreastCancer.py
```

## Output
The application generates:
- **Model Performance Metrics**: Accuracy percentage and detailed classification report
- **Visualizations**: Saved in `artifacts_breast_cancer/plots/`
  - `correlation_heatmap.png`: Feature correlation matrix
  - `confusion_matrix.png`: Model prediction accuracy matrix
  - `feature_importance.png`: SVM feature coefficients
- **Trained Model**: Saved in `artifacts_breast_cancer/models/` with timestamp

## Model Performance
The Linear SVM model typically achieves:
- **Accuracy**: 95%+ on test data
- **High Precision and Recall**: For both malignant and benign classes
- **Feature Importance**: Identifies most discriminative features for classification

## Key Insights
1. **Feature Selection**: SVM coefficients help identify most important features
2. **High Accuracy**: Linear SVM performs well on this linearly separable dataset
3. **Medical Relevance**: Model can assist in early cancer detection
4. **Interpretability**: Feature importance provides medical insights

## Dataset Features
The Wisconsin Breast Cancer dataset includes 30 features derived from digitized images of fine needle aspirate (FNA) of breast mass:

### Mean Features (10):
- radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean
- compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean

### Standard Error Features (10):
- radius_se, texture_se, perimeter_se, area_se, smoothness_se
- compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se

### Worst Features (10):
- radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst
- compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst

## File Structure
```
Breast_Cancer_Case_Study/
├── BreastCancer.py
├── requirements.txt
├── README.md
└── artifacts_breast_cancer/
    ├── models/
    │   └── breast_cancer_pipeline_*.joblib
    └── plots/
        ├── correlation_heatmap.png
        ├── confusion_matrix.png
        └── feature_importance.png
```

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- joblib >= 1.3.2

## Medical Context
This model can be used as a preliminary screening tool in medical diagnosis:
- **Benign (0)**: Non-cancerous tumor, typically requires monitoring
- **Malignant (1)**: Cancerous tumor, requires immediate medical attention
- **Early Detection**: Helps in identifying potential cancer cases early

## Author
**Harshad Kawade**  
*Date: 18/09/2025*

