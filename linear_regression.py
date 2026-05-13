import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

from data_preprocessing import load_and_explore_data, preprocess_data
import os
from utils import evaluate_model, predict_strength, save_model, OUTPUT_DIRS

def train_linear_regression(X_train, y_train, X_test, y_test):
    """
    Train a Linear Regression model and evaluate its performance.
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training target values
    X_test: Test features
    y_test: Test target values
    
    Returns:
    --------
    model: Trained Linear Regression model
    r2: R-squared score
    """
    print("\nTraining Linear Regression model...")
    lr = LinearRegression()
    lr_model, lr_pred, lr_mse, lr_r2 = evaluate_model(
        lr, X_train, X_test, y_train, y_test, "Linear Regression"
    )
    
    return lr_model, lr_r2

def main():
    # Load and preprocess data
    df = load_and_explore_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train Linear Regression model
    lr_model, lr_mse, lr_r2 = train_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save the model
    save_model(lr_model, os.path.join(OUTPUT_DIRS['models'], 'linear_regression_model.pkl'))
    
    # Example prediction
    new_concrete = {
        'cement': 400,
        'pofa': 100,
        'ca_fa': 1.30,
        'w_b': 0.35,
        'sp': 5.0,
        'age': 90
    }
    
    prediction = predict_strength(
        lr_model, scaler,
        new_concrete['cement'],
        new_concrete['pofa'],
        new_concrete['ca_fa'],
        new_concrete['w_b'],
        new_concrete['sp'],
        new_concrete['age']
    )
    
    print(f"\nLinear Regression Prediction for New Concrete: {prediction:.2f} MPa")

if __name__ == "__main__":
    main()