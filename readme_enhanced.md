# Enhanced POFA Concrete Strength Prediction

This project provides a comprehensive collection of machine learning models to predict the compressive strength of concrete containing Palm Oil Fuel Ash (POFA).

## Models Implemented

### Original Models

- Linear Regression
- Random Forest
- Gradient Boosting
- Support Vector Regression (SVR)
- Stacking Ensemble (Hybrid model)
- Bagging Algorithm
- Gene Expression Programming (GEP)

## File Structure

- `data_preprocessing.py` - Functions for loading, exploring, and preprocessing data
- `utils.py` - Utility functions shared across models
- `linear_regression.py` - Linear Regression model implementation
- `random_forest.py` - Random Forest model implementation
- `gradient_boosting.py` - Gradient Boosting model implementation
- `svr.py` - Support Vector Regression model implementation
- `hybrid_model.py` - Stacking Ensemble model implementation
- `ann_model.py` - Artificial Neural Network model implementation
- `bagging_algorithm.py` - Bagging algorithm implementation
- `gep_model.py` - Gene Expression Programming implementation (using DEAP library)
- `model_comparison.py` - Functions to compare all model performances
- `correlation_analysis.py` - Pearson correlation analysis for model predictions
- `main_enhanced.py` - Enhanced main script to train models and make predictions
- `main.py` - Legacy main script (for backward compatibility)

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- TensorFlow (for ANN model)
- DEAP (for GEP model)
- SciPy
- SymPy (for GEP model formula visualization)

Install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

### Training Models

To train all models and save them:

```bash
python main_enhanced.py --model all --data pofa_concrete_data.csv
```

To train a specific model:

```bash
python main_enhanced.py --model ann  # Options: linear, rf, gb, svr, hybrid, ann, bagging, gep
```

### Making Predictions

To predict using a trained model:

```bash
python main_enhanced.py --predict --model ann --cement 400 --pofa 100 --ca_fa 1.3 --w_b 0.35 --sp 5.0 --age 28
```

### Comparing Models

To compare all models:

```bash
python main_enhanced.py --model compare
```

### Correlation Analysis

To generate Pearson correlation analysis between predicted and actual values:

```bash
python main_enhanced.py --model correlation
```

## Model Parameters

When making predictions, you can specify the following parameters:

- `--cement`: Cement content (kg/m³)
- `--pofa`: Palm Oil Fuel Ash content (kg/m³)
- `--ca_fa`: Coarse Aggregate to Fine Aggregate ratio
- `--w_b`: Water to Binder ratio
- `--sp`: Superplasticizer content (kg/m³)
- `--age`: Age of concrete (days)

## Output Files

The system generates several output files:

### Model Files

- `linear_regression_model.pkl` - Trained Linear Regression model
- `random_forest_model_tuned.pkl` - Tuned Random Forest model
- `gradient_boosting_model_tuned.pkl` - Tuned Gradient Boosting model
- `svr_model_tuned.pkl` - Tuned SVR model
- `hybrid_model.pkl` - Hybrid model
- `ann_model.h5` - Trained ANN model
- `bagging_model_tuned.pkl` - Tuned Bagging model
- `gep_model.pkl` - GEP model and formula
- `pofa_concrete_strength_scaler.pkl` - Feature scaler

### Visualization Files

- Various model performance plots (e.g., actual vs. predicted)
- `correlation_matrix.png` - Pearson correlation matrix for all models
- `individual_correlations.png` - Individual correlation plots
- `best_model_correlation.png` - Detailed correlation plot for the best model

## Example Usage

```python
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Load model and scaler
model = load_model('ann_model.h5')  # or any other model
scaler = joblib.load('pofa_concrete_strength_scaler.pkl')

# Create input data
input_data = np.array([
    [400, 100, 1.3, 0.35, 5.0, 28]  # cement, pofa, ca_fa, w_b, sp, age
])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = float(model.predict(input_data_scaled)[0][0])
print(f"Predicted Compressive Strength: {prediction:.2f} MPa")
```
