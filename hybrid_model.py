import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from data_preprocessing import load_and_explore_data, preprocess_data
from utils import predict_strength, save_model

def train_hybrid_model(X_train, y_train, X_test, y_test):
    """
    Train a Stacking Ensemble (hybrid) model and evaluate its performance.
    
    Parameters:
    -----------
    X_train: Training features
    y_train: Training target values
    X_test: Test features
    y_test: Test target values
    
    Returns:
    --------
    model: Trained hybrid model
    r2: R-squared score
    """
    print("\nTraining Hybrid (Stacking Ensemble) model...")
    
    # Define base estimators
    estimators = [
        ('rf', RandomForestRegressor(random_state=42)),
        ('gb', GradientBoostingRegressor(random_state=42))
    ]
    
    # Create stacking ensemble model
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=SVR()
    )
    
    # Cross-validation for the hybrid model
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(stacking_model, X_train, y_train, cv=kf, scoring='r2')
    
    print("Hybrid Model (Stacking) Cross-Validation Results:")
    print(f"Average R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    # Final evaluation of the hybrid model
    stacking_model.fit(X_train, y_train)
    y_pred_hybrid = stacking_model.predict(X_test)
    
    mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
    rmse_hybrid = np.sqrt(mse_hybrid)
    r2_hybrid = r2_score(y_test, y_pred_hybrid)
    mae_hybrid = mean_absolute_error(y_test, y_pred_hybrid)
    
    print(f"Hybrid Model Performance:")
    print(f"MSE: {mse_hybrid:.4f}")
    print(f"RMSE: {rmse_hybrid:.4f}")
    print(f"R²: {r2_hybrid:.4f}")
    print(f"MAE: {mae_hybrid:.4f}")
    
    return stacking_model, r2_hybrid

def main():
    # Load and preprocess data
    df = load_and_explore_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train hybrid model
    hybrid_model, hybrid_r2 = train_hybrid_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Save the hybrid model
    save_model(hybrid_model, 'hybrid_model.pkl')
    
    # Example prediction with hybrid model
    new_concrete = {
        'cement': 400,
        'pofa': 100,
        'ca_fa': 1.30,
        'w_b': 0.35,
        'sp': 5.0,
        'age': 90
    }
    
    prediction = predict_strength(
        hybrid_model, scaler,
        new_concrete['cement'],
        new_concrete['pofa'],
        new_concrete['ca_fa'],
        new_concrete['w_b'],
        new_concrete['sp'],
        new_concrete['age']
    )
    
    print(f"\nHybrid Model Prediction for New Concrete: {prediction:.2f} MPa")

if __name__ == "__main__":
    main()