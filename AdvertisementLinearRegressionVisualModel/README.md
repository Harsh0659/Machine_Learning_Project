# Advertisement Sales Prediction Case Study

## Overview
This case study focuses on predicting sales based on advertising spending across different media channels (TV, Radio, and Newspaper). The project uses Linear Regression to build a predictive model and provides comprehensive analysis through data visualization and model evaluation.

## Problem Statement
Predict sales revenue based on advertising expenditure in different media channels to help businesses optimize their advertising budget allocation.

## Dataset
- **File**: `Advertising.csv`
- **Features**: 
  - TV: Advertising spending on TV (in thousands)
  - Radio: Advertising spending on Radio (in thousands)
  - Newspaper: Advertising spending on Newspaper (in thousands)
- **Target**: Sales (in thousands of units)
- **Size**: 200 records

## Features
- **Data Analysis**: Comprehensive exploratory data analysis with statistical summaries
- **Visualization**: 
  - Correlation heatmap
  - Pairplot for feature relationships
  - Actual vs Predicted scatter plot
  - Residual analysis plots
  - Feature importance visualization
- **Model**: Linear Regression with StandardScaler preprocessing
- **Evaluation**: MSE, RMSE, and R² metrics
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: Linear Regression
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
python AdvertisementLinearRegressionVisualModel.py --data Advertising.csv
```

### Command Line Arguments
- `--data`: Path to the dataset CSV file (default: Advertising.csv)

## Output
The application generates:
- **Model Performance Metrics**: MSE, RMSE, R² score
- **Visualizations**: Saved in `artifacts/plots/`
  - `correlation_heatmap.png`: Feature correlation matrix
  - `pairplot.png`: Pairwise feature relationships
  - `actual_vs_predicted.png`: Model prediction accuracy
  - `residual_plot.png`: Residual analysis
  - `residual_distribution.png`: Residual distribution
  - `feature_importance.png`: Feature coefficients
- **Trained Model**: Saved in `artifacts/models/` with timestamp

## Model Performance
The Linear Regression model typically achieves:
- **R² Score**: ~0.90+ (90%+ variance explained)
- **Low RMSE**: Indicating good prediction accuracy
- **Feature Importance**: TV advertising typically shows highest impact

## Key Insights
1. **TV Advertising** usually has the strongest positive correlation with sales
2. **Radio Advertising** shows moderate positive correlation
3. **Newspaper Advertising** often has the weakest correlation
4. The model provides interpretable coefficients for business decision-making

## File Structure
```
Advertisement_Case_Study/
├── AdvertisementLinearRegressionVisualModel.py
├── Advertising.csv
├── requirements.txt
├── README.md
└── artifacts/
    ├── models/
    │   └── marvellous_advertisement_model_*.pkl
    └── plots/
        ├── correlation_heatmap.png
        ├── pairplot.png
        ├── actual_vs_predicted.png
        ├── residual_plot.png
        ├── residual_distribution.png
        └── feature_importance.png
```

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- joblib >= 1.3.2

## Author
** Harshad Kawade**  
*Date: 18/09/2025*
