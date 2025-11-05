# Wine Quality Classification Case Study

## Overview
This case study focuses on wine quality classification using K-Nearest Neighbors (KNN) algorithm. The project analyzes various chemical properties of wine to predict its quality class, providing insights into wine production and quality assessment.

## Problem Statement
Classify wine quality based on various chemical properties and measurements to assist in wine production quality control and help consumers make informed purchasing decisions.

## Dataset
- **File**: `WinePredictor.csv`
- **Features**: 13 chemical properties of wine
  - Alcohol: Alcohol content percentage
  - Malic acid: Malic acid content
  - Ash: Ash content
  - Alkalinity of ash: Alkalinity of ash
  - Magnesium: Magnesium content
  - Total phenols: Total phenols content
  - Flavanoids: Flavanoids content
  - Nonflavanoid phenols: Nonflavanoid phenols content
  - Proanthocyanins: Proanthocyanins content
  - Color intensity: Color intensity
  - Hue: Hue measurement
  - OD280/OD315: OD280/OD315 of diluted wines
  - Proline: Proline content
- **Target**: Class (1, 2, 3 - Wine quality classes)
- **Size**: 178 samples

## Features
- **Data Preprocessing**: 
  - Missing value handling (dropna)
  - Feature scaling with StandardScaler
- **Hyperparameter Tuning**: 
  - K-value optimization (1-19)
  - Accuracy vs K plot for optimal parameter selection
- **Visualization**: 
  - Accuracy vs K value plot
- **Model**: K-Nearest Neighbors Classifier with optimal K
- **Evaluation**: Accuracy, Confusion Matrix, Classification Report
- **Artifacts**: Automated model saving with timestamps

## Technical Implementation
- **Algorithm**: K-Nearest Neighbors (KNN)
- **Preprocessing**: StandardScaler for feature normalization
- **Hyperparameter Tuning**: Grid search for optimal K value (1-19)
- **Validation**: 80/20 train-test split with random state for reproducibility
- **Distance Metric**: Euclidean distance (default)

## Usage

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application
```bash
python WinePredictorParameterTuningvisual.py --data WinePredictor.csv
```

### Command Line Arguments
- `--data`: Path to the dataset CSV file (default: WinePredictor.csv)

## Output
The application generates:
- **Model Performance Metrics**: Accuracy percentage, confusion matrix, and detailed classification report
- **Visualizations**: Saved in `artifacts_wine/plots/`
  - `accuracy_vs_k.png`: Plot showing accuracy for different K values
- **Trained Model**: Saved in `artifacts_wine/models/` with timestamp

## Model Performance
The KNN model typically achieves:
- **Accuracy**: 90-95% on test data with optimal K value
- **Good Precision and Recall**: For all wine quality classes
- **Optimal K**: Usually between 3-7 neighbors

## Key Insights
1. **Chemical Properties**: Certain chemical compounds strongly influence wine quality
2. **Alcohol Content**: Higher alcohol content often correlates with better quality
3. **Phenolic Compounds**: Flavanoids and total phenols are important quality indicators
4. **Color Properties**: Hue and color intensity affect quality classification
5. **KNN Effectiveness**: KNN performs well on this relatively small, well-separated dataset

## Wine Quality Classes
The dataset contains three wine quality classes:
- **Class 1**: Lower quality wines
- **Class 2**: Medium quality wines  
- **Class 3**: Higher quality wines

## Dataset Features Description
### Chemical Properties:
- **Alcohol**: Alcohol content percentage (11.03-14.83)
- **Malic acid**: Malic acid content (0.74-5.80)
- **Ash**: Ash content (1.36-3.23)
- **Alkalinity of ash**: Alkalinity of ash (10.6-30.0)
- **Magnesium**: Magnesium content (70-162)
- **Total phenols**: Total phenols content (0.98-3.88)
- **Flavanoids**: Flavanoids content (0.34-5.08)
- **Nonflavanoid phenols**: Nonflavanoid phenols content (0.13-0.66)
- **Proanthocyanins**: Proanthocyanins content (0.41-3.58)
- **Color intensity**: Color intensity (1.28-13.0)
- **Hue**: Hue measurement (0.48-1.71)
- **OD280/OD315**: OD280/OD315 of diluted wines (1.27-4.00)
- **Proline**: Proline content (278-1680)

## KNN Algorithm Benefits
- **Non-parametric**: No assumptions about data distribution
- **Simple**: Easy to understand and implement
- **Effective**: Works well on small to medium datasets
- **Interpretable**: Results are easy to explain
- **Robust**: Handles noise well

## Hyperparameter Tuning Process
1. **K Range**: Test K values from 1 to 19
2. **Cross-validation**: Use train-test split for evaluation
3. **Accuracy Plot**: Visualize performance across K values
4. **Optimal Selection**: Choose K with highest accuracy
5. **Final Model**: Train with optimal K value

## File Structure
```
Wine_Predictor_Case_Study/
├── WinePredictorParameterTuningvisual.py
├── WinePredictor.csv
├── requirements.txt
├── README.md
└── artifacts_wine/
    ├── models/
    │   └── wine_knn_model_*.joblib
    └── plots/
        └── accuracy_vs_k.png
```

## Wine Industry Applications
This model can be used for:
- **Quality Control**: Automated wine quality assessment
- **Production Optimization**: Identify key factors for better wine
- **Consumer Guidance**: Help consumers choose quality wines
- **Research**: Study chemical properties affecting wine quality
- **Competition Judging**: Assist in wine competition evaluations

## Wine Quality Factors
Key factors influencing wine quality:
1. **Alcohol Content**: Balance is crucial for quality
2. **Acidity**: Malic acid affects taste and preservation
3. **Phenolic Compounds**: Contribute to color, taste, and health benefits
4. **Color Properties**: Visual appeal and quality indicators
5. **Mineral Content**: Ash and magnesium affect taste profile

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

