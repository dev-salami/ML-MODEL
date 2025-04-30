import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

from data_preprocessing import load_and_explore_data, preprocess_data
from utils import evaluate_model, predict_strength, save_model

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Train a Random Forest model and evaluate its performance.
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training target values
    X_test: Test features
    y_test: Test target values
    
    Returns:
    --------
    model: Trained Random Forest model
    r2: R-squared score
    """
    print("\nTraining Random Forest model...")
    rf = RandomForestRegressor(random_state=42)
    rf_model, rf_pred, rf_mse, rf_r2 = evaluate_model(
        rf, X_train, X_test, y_train, y_test, "Random Forest"
    )
    
    return rf_model, rf_r2

def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Tune hyperparameters for Random Forest model using GridSearchCV.
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training target values
    X_test: Test features
    y_test: Test target values
    
    Returns:
    --------
    tuned_model: Tuned Random Forest model
    r2_tuned: R-squared score of the tuned model
    """
    print("\nTuning Random Forest model...")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Base model
    rf = RandomForestRegressor(random_state=42)
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='r2')
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
    
    print(f"Tuned Random Forest Performance:")
    print(f"MSE: {mse_tuned:.4f}")
    print(f"RMSE: {np.sqrt(mse_tuned):.4f}")
    print(f"R²: {r2_tuned:.4f}")
    
    return tuned_model, r2_tuned

def main():
    # Load and preprocess data
    df = load_and_explore_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train base Random Forest model
    rf_model, rf_mse, rf_r2 = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Tune Random Forest model
    tuned_rf_model, tuned_rf_r2 = tune_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save the tuned model
    save_model(tuned_rf_model, 'random_forest_model_tuned.pkl')
    
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
        tuned_rf_model, scaler,
        new_concrete['cement'],
        new_concrete['pofa'],
        new_concrete['ca_fa'],
        new_concrete['w_b'],
        new_concrete['sp'],
        new_concrete['age']
    )
    
    print(f"\nTuned Random Forest Prediction for New Concrete: {prediction:.2f} MPa")

if __name__ == "__main__":
    main()