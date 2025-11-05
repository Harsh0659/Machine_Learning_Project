# Telco Customer Churn Prediction Case Study

## Overview
This case study focuses on predicting customer churn in telecommunications using Random Forest Classifier. The project analyzes customer behavior patterns and service usage to identify customers likely to leave the service, enabling proactive retention strategies.

## Problem Statement
Predict which customers are likely to churn (leave the service) based on their demographic information, account details, and service usage patterns to implement targeted retention strategies and reduce customer loss.

## Dataset
- **File**: `Telco-Customer-Churn.csv`
- **Features**: 
  - Demographics: Gender, SeniorCitizen, Partner, Dependents
  - Account Info: Tenure, Contract, PaperlessBilling, PaymentMethod
  - Services: PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
  - Charges: MonthlyCharges, TotalCharges
- **Target**: Churn (Yes/No)
- **Size**: 7,043 customers

## Features
- **Data Preprocessing**: 
  - Categorical feature encoding using LabelEncoder
  - Removal of irrelevant columns (customerID, gender)
  - Feature scaling with StandardScaler
- **Visualization**: 
  - Feature importance analysis from Random Forest
- **Model**: Random Forest Classifier with optimized parameters
- **Evaluation**: Accuracy, Classification Report
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: Random Forest Classifier
- **Preprocessing**: LabelEncoder for categorical variables, StandardScaler for feature normalization
- **Pipeline**: Scikit-learn Pipeline for streamlined workflow
- **Validation**: 80/20 train-test split with random state for reproducibility
- **Hyperparameters**: n_estimators=150, max_depth=7

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python RandomForestFetureImportance.py --data Telco-Customer-Churn.csv
```

### Command Line Arguments
- `--data`: Path to the dataset CSV file (default: Telco-Customer-Churn.csv)

## Output
The application generates:
- **Model Performance Metrics**: Accuracy percentage and detailed classification report
- **Visualizations**: Saved in `artifacts_telco/plots/`
  - `feature_importance.png`: Random Forest feature importance ranking
- **Trained Model**: Saved in `artifacts_telco/models/` with timestamp

## Model Performance
The Random Forest model typically achieves:
- **Accuracy**: 80-85% on test data
- **Good Precision and Recall**: For both churn and non-churn classes
- **Feature Importance**: Identifies key factors driving customer churn

## Key Insights
1. **Contract Type**: Month-to-month contracts have higher churn rates
2. **Tenure**: Longer tenure customers are less likely to churn
3. **Internet Service**: Fiber optic customers show higher churn rates
4. **Payment Method**: Electronic check users have higher churn rates
5. **Monthly Charges**: Higher charges correlate with increased churn

## Feature Importance Analysis
The Random Forest model typically identifies these as most important:
1. **Total Charges**: Cumulative amount paid by customer
2. **Tenure**: Length of customer relationship
3. **Contract Type**: Contract duration and terms
4. **Monthly Charges**: Regular payment amount
5. **Internet Service Type**: Type of internet connection

## Business Applications
This model can be used for:
- **Customer Retention**: Identify high-risk customers for targeted retention campaigns
- **Marketing Strategy**: Develop personalized offers for at-risk customers
- **Resource Allocation**: Focus retention efforts on customers most likely to churn
- **Service Improvement**: Address issues causing customer dissatisfaction

## Dataset Features Description
### Demographics:
- **SeniorCitizen**: Whether customer is 65 or older (0/1)
- **Partner**: Whether customer has a partner (Yes/No)
- **Dependents**: Whether customer has dependents (Yes/No)

### Account Information:
- **Tenure**: Number of months customer has been with company
- **Contract**: Contract type (Month-to-month, One year, Two year)
- **PaperlessBilling**: Whether customer has paperless billing (Yes/No)
- **PaymentMethod**: Payment method used

### Services:
- **PhoneService**: Whether customer has phone service (Yes/No)
- **MultipleLines**: Whether customer has multiple lines (Yes/No/No phone service)
- **InternetService**: Type of internet service (DSL, Fiber optic, No)
- **OnlineSecurity**: Whether customer has online security (Yes/No/No internet service)
- **OnlineBackup**: Whether customer has online backup (Yes/No/No internet service)
- **DeviceProtection**: Whether customer has device protection (Yes/No/No internet service)
- **TechSupport**: Whether customer has tech support (Yes/No/No internet service)
- **StreamingTV**: Whether customer has streaming TV (Yes/No/No internet service)
- **StreamingMovies**: Whether customer has streaming movies (Yes/No/No internet service)

### Charges:
- **MonthlyCharges**: Monthly amount charged to customer
- **TotalCharges**: Total amount charged to customer

## File Structure
```
Telco_Case_Study/
├── RandomForestFetureImportance.py
├── Telco-Customer-Churn.csv
├── requirements.txt
├── README.md
└── artifacts_telco/
    ├── models/
    │   └── telco_randomforest_pipeline_*.joblib
    └── plots/
        └── feature_importance.png
```

## Churn Prevention Strategies
Based on model insights, effective strategies include:
1. **Contract Incentives**: Offer discounts for longer-term contracts
2. **Service Bundles**: Create attractive service packages
3. **Payment Options**: Provide flexible payment methods
4. **Customer Support**: Enhance technical support services
5. **Loyalty Programs**: Implement retention rewards

## Dependencies
- pandas >= 2.1.0
- numpy >= 1.25.0
- matplotlib >= 3.8.0
- seaborn >= 0.12.2
- scikit-learn >= 1.3.0
- joblib >= 1.3.2

## Author
**Harshad Kawade**  
*Date: 18/09/2025*

