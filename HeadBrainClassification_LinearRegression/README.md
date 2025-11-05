# Head-Brain Size Prediction Case Study

## Overview
This case study focuses on predicting brain weight based on head size using Linear Regression. The project analyzes the relationship between head circumference and brain weight, providing insights into human anatomy and potential medical applications.

## Problem Statement
Predict brain weight (in grams) based on head size measurements (in cm³) to understand the correlation between head size and brain weight, which can be useful in medical research and anthropological studies.

## Dataset
- **File**: `MarvellousHeadBrain.csv`
- **Features**: 
  - Head Size(cm^3): Head circumference measurement in cubic centimeters
- **Target**: Brain Weight(grams): Brain weight in grams
- **Size**: 237 records

## Features
- **Data Analysis**: Comprehensive exploratory data analysis with statistical summaries
- **Visualization**: 
  - Regression line plot showing the relationship between head size and brain weight
  - Scatter plot with actual vs predicted values
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
python HeadBrainVisualResult.py --data MarvellousHeadBrain.csv
```

### Command Line Arguments
- `--data`: Path to the dataset CSV file (default: MarvellousHeadBrain.csv)

## Output
The application generates:
- **Model Performance Metrics**: MSE, RMSE, and R² score
- **Visualizations**: Saved in `artifacts_head_brain/plots/`
  - `regression_line.png`: Scatter plot with regression line showing head size vs brain weight relationship
- **Trained Model**: Saved in `artifacts_head_brain/models/` with timestamp

## Model Performance
The Linear Regression model typically achieves:
- **R² Score**: 0.60-0.80 (60-80% variance explained)
- **Low RMSE**: Indicating reasonable prediction accuracy
- **Linear Relationship**: Strong positive correlation between head size and brain weight

## Key Insights
1. **Positive Correlation**: Head size and brain weight show strong positive linear relationship
2. **Medical Relevance**: Can be useful in medical imaging and brain development studies
3. **Anthropological Value**: Provides insights into human brain size variations
4. **Simple Model**: Linear regression captures the relationship effectively

## Scientific Context
This model can be used for:
- **Medical Research**: Understanding brain development patterns
- **Anthropological Studies**: Analyzing human brain size variations
- **Medical Imaging**: Estimating brain weight from head measurements
- **Developmental Studies**: Tracking brain growth patterns

## Dataset Characteristics
- **Head Size Range**: Typically 2500-5500 cm³
- **Brain Weight Range**: Typically 1000-2000 grams
- **Sample Size**: 237 individuals
- **Measurement Units**: 
  - Head Size: Cubic centimeters (cm³)
  - Brain Weight: Grams (g)

## Regression Analysis
The model provides:
- **Slope (m)**: Rate of change in brain weight per unit increase in head size
- **Intercept (c)**: Predicted brain weight when head size is zero (theoretical)
- **Equation**: Brain Weight = m × Head Size + c

## File Structure
```
Head_Brain_Case_Study/
├── HeadBrainVisualResult.py
├── MarvellousHeadBrain.csv
├── requirements.txt
├── README.md
└── artifacts_head_brain/
    ├── models/
    │   └── head_brain_pipeline_*.joblib
    └── plots/
        └── regression_line.png
```

## Medical Applications
This model can be applied in:
- **Neurosurgery**: Pre-operative planning and assessment
- **Pediatric Medicine**: Monitoring brain development in children
- **Geriatric Care**: Understanding age-related brain changes
- **Research**: Studying brain size variations across populations

## Limitations
- **Simplified Model**: Real brain weight depends on many factors beyond head size
- **Individual Variation**: Significant variation exists between individuals
- **Age Factor**: Brain weight changes with age (not considered in this model)
- **Gender Differences**: Brain size varies between genders (not explicitly modeled)

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
