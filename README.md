# POFA Concrete Strength Prediction

This project provides a collection of machine learning models to predict the compressive strength of concrete containing Palm Oil Fuel Ash (POFA).

## File Structure

- `data_preprocessing.py` - Functions for loading, exploring, and preprocessing data
- `utils.py` - Utility functions shared across models
- `linear_regression.py` - Linear Regression model implementation
- `random_forest.py` - Random Forest model implementation
- `gradient_boosting.py` - Gradient Boosting model implementation
- `svr.py` - Support Vector Regression model implementation
- `hybrid_model.py` - Stacking Ensemble model implementation
- `model_comparison.py` - Functions to compare all model performances
- `main.py` - Main script to train models and make predictions

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib

## Usage

### Training Models

To train all models and save them:

```bash
python main.py --model all --data pofa_concrete_data.csv
```

To train a specific model:

```bash
python3 main.py --model rf  # Options: linear, rf, gb, svr, hybrid
```

### Making Predictions

To predict using a trained model:

```bash
python3 main.py --predict --model rf --cement 400 --pofa 100 --ca_fa 1.3 --w_b 0.35 --sp 5.0 --age 28
```

### Comparing Models

To compare all models:

```bash
python3 main.py --model compare
```

## Model Parameters

When making predictions, you can specify the following parameters:

- `--cement`: Cement content (kg/m³)
- `--pofa`: Palm Oil Fuel Ash content (kg/m³)
- `--ca_fa`: Coarse Aggregate to Fine Aggregate ratio
- `--w_b`: Water to Binder ratio
- `--sp`: Superplasticizer content (kg/m³)
- `--age`: Age of concrete (days)

## Notes about the `predict_strength` Function

The `predict_strength` function in `utils.py` has been designed to handle different input scenarios:

1. If the scaler was trained with 7 features (including the target variable), the function will automatically add a dummy value (0) for the 7th feature to match the expected input dimension.
2. If the scaler was trained with just the 6 input features, the function will use only those 6 features.

This design ensures compatibility regardless of how the scaler was trained.

## Example Usage

```python
import joblib
from utils import predict_strength

# Load model and scaler
model = joblib.load('random_forest_model_tuned.pkl')
scaler = joblib.load('pofa_concrete_strength_scaler.pkl')

# Make prediction
strength = predict_strength(
    model, scaler,
    cement=400,
    pofa=100,
    ca_fa=1.3,
    w_b=0.35,
    sp=5.0,
    age=28
)

print(f"Predicted Compressive Strength: {strength:.2f} MPa")
```
