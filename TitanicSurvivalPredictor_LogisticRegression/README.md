# Titanic Survival Prediction Case Study

## Overview
This case study focuses on predicting passenger survival on the Titanic using Logistic Regression. The project analyzes various passenger characteristics and circumstances to predict who survived the tragic sinking, providing insights into survival patterns and historical analysis.

## Problem Statement
Predict passenger survival on the Titanic based on demographic information, ticket class, fare, and other factors to understand survival patterns and create a memorial analysis of this historical event.

## Dataset
- **File**: `MarvellousTitanicDataset.csv`
- **Features**: 
  - Pclass: Passenger class (1st, 2nd, 3rd)
  - Name: Passenger name
  - Sex: Gender (male/female)
  - Age: Age in years
  - SibSp: Number of siblings/spouses aboard
  - Parch: Number of parents/children aboard
  - Ticket: Ticket number
  - Fare: Passenger fare
  - Cabin: Cabin number
  - Embarked: Port of embarkation (C, Q, S)
- **Target**: Survived (0: Did not survive, 1: Survived)
- **Size**: 891 passengers

## Features
- **Data Preprocessing**: 
  - Categorical feature encoding using LabelEncoder
  - Missing value handling with median imputation
  - Removal of irrelevant columns (PassengerId, zero)
- **Visualization**: 
  - Survival distribution analysis
  - Survival by gender analysis
  - Survival by passenger class analysis
  - Correlation heatmap
- **Model**: Logistic Regression with StandardScaler preprocessing
- **Evaluation**: Accuracy, Confusion Matrix, Classification Report
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: Logistic Regression
- **Preprocessing**: LabelEncoder for categorical variables, StandardScaler for feature normalization
- **Pipeline**: Scikit-learn Pipeline for streamlined workflow
- **Validation**: 80/20 train-test split with stratified sampling for balanced classes
- **Missing Values**: Median imputation for numerical features

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python Titianicvisualfinal.py --data MarvellousTitanicDataset.csv
```

### Command Line Arguments
- `--data`: Path to the dataset CSV file (default: MarvellousTitanicDataset.csv)

## Output
The application generates:
- **Model Performance Metrics**: Accuracy percentage, confusion matrix, and detailed classification report
- **Visualizations**: Saved in `artifacts_titanic/plots/`
  - `survival_distribution.png`: Overall survival rate distribution
  - `survival_by_gender.png`: Survival rates by gender
  - `survival_by_class.png`: Survival rates by passenger class
  - `correlation_heatmap.png`: Feature correlation matrix
- **Trained Model**: Saved in `artifacts_titanic/models/` with timestamp

## Model Performance
The Logistic Regression model typically achieves:
- **Accuracy**: 80-85% on test data
- **Good Precision and Recall**: For both survival and non-survival classes
- **Feature Importance**: Identifies key factors affecting survival

## Key Insights
1. **Gender**: Women had significantly higher survival rates than men
2. **Passenger Class**: First-class passengers had better survival chances
3. **Age**: Children had higher survival rates
4. **Fare**: Higher fare passengers had better survival chances
5. **Family Size**: Moderate family size (SibSp + Parch) was optimal

## Historical Context
The Titanic disaster (April 15, 1912) provides insights into:
- **Social Class Impact**: How socioeconomic status affected survival
- **Gender Roles**: "Women and children first" policy implementation
- **Lifeboat Capacity**: Limited lifeboats affected survival rates
- **Geographic Factors**: Location on ship influenced survival chances

## Dataset Features Description
### Demographics:
- **Pclass**: Passenger class (1 = 1st class, 2 = 2nd class, 3 = 3rd class)
- **Name**: Passenger name
- **Sex**: Gender (male/female)
- **Age**: Age in years (0.42-80)

### Family Information:
- **SibSp**: Number of siblings/spouses aboard (0-8)
- **Parch**: Number of parents/children aboard (0-6)

### Travel Details:
- **Ticket**: Ticket number
- **Fare**: Passenger fare (0-512.33)
- **Cabin**: Cabin number
- **Embarked**: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)

## Survival Patterns
Historical analysis reveals:
- **Overall Survival Rate**: ~38% of passengers survived
- **Women's Survival Rate**: ~74% of women survived
- **Men's Survival Rate**: ~19% of men survived
- **Children's Survival Rate**: ~59% of children survived
- **Class Impact**: 1st class (63%), 2nd class (47%), 3rd class (24%)

## File Structure
```
Titanic_Case_Study/
├── Titianicvisualfinal.py
├── MarvellousTitanicDataset.csv
├── requirements.txt
├── README.md
├── saved_models/
│   ├── titanic_survival_model_*.pkl
│   └── titanic_survival_model.pkl
└── artifacts_titanic/
    ├── models/
    │   └── titanic_pipeline_*.joblib
    └── plots/
        ├── survival_distribution.png
        ├── survival_by_gender.png
        ├── survival_by_class.png
        └── correlation_heatmap.png
```

## Memorial and Educational Value
This project serves as:
- **Historical Memorial**: Honoring the victims and survivors of the Titanic
- **Educational Tool**: Teaching data science through historical context
- **Social Analysis**: Understanding historical social dynamics
- **Data Science Practice**: Real-world application of machine learning

## Ethical Considerations
- **Respectful Analysis**: Treating the data with historical sensitivity
- **Educational Purpose**: Using the tragedy for learning and remembrance
- **Memorial Value**: Honoring the memory of those who perished
- **Historical Accuracy**: Maintaining factual integrity in analysis

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



## In Memory
This project is dedicated to the memory of all those who perished in the Titanic disaster on April 15, 1912. May their memory be a blessing and may this analysis serve as a reminder of the importance of safety, equality, and human dignity.
