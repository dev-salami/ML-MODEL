import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import joblib

from data_preprocessing import load_and_explore_data, preprocess_data
import os
from utils import evaluate_model, predict_strength, save_model, OUTPUT_DIRS

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Train a Gradient Boosting model and evaluate its performance.
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training target values
    X_test: Test features
    y_test: Test target values
    
    Returns:
    --------
    model: Trained Gradient Boosting model
    r2: R-squared score
    """
    print("\nTraining Gradient Boosting model...")
    gb = GradientBoostingRegressor(random_state=42)
    gb_model, gb_pred, gb_mse, gb_r2 = evaluate_model(
        gb, X_train, X_test, y_train, y_test, "Gradient Boosting"
    )
    
    return gb_model, gb_r2

def tune_gradient_boosting(X_train, y_train, X_test, y_test):
    """
    Tune hyperparameters for Gradient Boosting model using GridSearchCV.
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training target values
    X_test: Test features
    y_test: Test target values
    
    Returns:
    --------
    tuned_model: Tuned Gradient Boosting model
    r2_tuned: R-squared score of the tuned model
    """
    print("\nTuning Gradient Boosting model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    
    # Base model
    gb = GradientBoostingRegressor(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='r2')
    grid_search.fit(X_train, y_train)
    
    # Best hyperparameters
    print("Best Hyperparameters:")
    print(grid_search.best_params_)
    
    # Get the best model
    tuned_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred_tuned = tuned_model.predict(X_test)
    mse_tuned = ((y_test - y_pred_tuned) ** 2).mean()
    r2_tuned = 1 - (((y_test - y_pred_tuned) ** 2).sum() / 
                     ((y_test - y_test.mean()) ** 2).sum())
    
    print(f"Tuned Gradient Boosting Performance:")
    print(f"MSE: {mse_tuned:.4f}")
    print(f"RMSE: {np.sqrt(mse_tuned):.4f}")
    print(f"R²: {r2_tuned:.4f}")
    
    return tuned_model, r2_tuned

def main():
    # Load and preprocess data
    df = load_and_explore_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train base Gradient Boosting model
    gb_model, gb_mse, gb_r2 = train_gradient_boosting(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Tune Gradient Boosting model
    tuned_gb_model, tuned_gb_r2 = tune_gradient_boosting(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save the tuned model
    save_model(tuned_gb_model, os.path.join(OUTPUT_DIRS['models'], 'gradient_boosting_model_tuned.pkl'))
    
    # Example prediction with tuned model
    new_concrete = {
        'cement': 400,
        'pofa': 100,
        'ca_fa': 1.30,
        'w_b': 0.35,
        'sp': 5.0,
        'age': 90
    }
    
    prediction = predict_strength(
        tuned_gb_model, scaler,
        new_concrete['cement'],
        new_concrete['pofa'],
        new_concrete['ca_fa'],
        new_concrete['w_b'],
        new_concrete['sp'],
        new_concrete['age']
    )
    
    print(f"\nTuned Gradient Boosting Prediction for New Concrete: {prediction:.2f} MPa")

if __name__ == "__main__":
    main()