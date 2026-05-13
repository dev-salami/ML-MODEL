import os
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR

from data_preprocessing import load_and_explore_data, preprocess_data
from utils import evaluate_model, predict_strength, OUTPUT_DIRS

# Import individual model modules
import linear_regression
import random_forest
import gradient_boosting
import svr
import hybrid_model

def compare_all_models():
    """
    Train all models and compare their performance.
    """
    # Load and preprocess data
    df = load_and_explore_data()
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Train all models
    print("Training and comparing all models...")
    
    # Linear Regression
    lr_model, lr_r2 = linear_regression.train_linear_regression(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Random Forest
    rf_model, rf_r2 = random_forest.train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Gradient Boosting
    gb_model, gb_r2 = gradient_boosting.train_gradient_boosting(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # SVR
    svr_model, svr_r2 = svr.train_svr(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Tune the best model (assuming Random Forest for simplicity)
    tuned_rf_model, tuned_rf_r2 = random_forest.tune_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Hybrid model
    hybrid_model_trained, hybrid_r2 = hybrid_model.train_hybrid_model(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Visualize model comparison
    models = ['Linear Regression', 'Random Forest', 'Gradient Boosting', 'SVR', 'Tuned RF', 'Hybrid']
    r2_scores = [lr_r2, rf_r2, gb_r2, svr_r2, tuned_rf_r2, hybrid_r2]
    
    plt.figure(figsize=(12, 6))
    plt.bar(models, r2_scores)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Comparison - R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS['misc'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['misc'], 'model_comparison.png'))
    plt.close()
    
    # Find the best model
    best_score = max(r2_scores)
    best_index = r2_scores.index(best_score)
    best_model_name = models[best_index]
    
    print(f"\nBest Model: {best_model_name} with R² score of {best_score:.4f}")
    
    # Example prediction with all models
    new_concrete = {
        'cement': 400,
        'pofa': 100,
        'ca_fa': 1.30,
        'w_b': 0.35,
        'sp': 5.0,
        'age': 90
    }
    
    print("\nPredictions for New Concrete Sample:")
    print(f"Linear Regression: {predict_strength(lr_model, scaler, new_concrete['cement'], new_concrete['pofa'], new_concrete['ca_fa'], new_concrete['w_b'], new_concrete['sp'], new_concrete['age']):.2f} MPa")
    print(f"Random Forest: {predict_strength(rf_model, scaler, new_concrete['cement'], new_concrete['pofa'], new_concrete['ca_fa'], new_concrete['w_b'], new_concrete['sp'], new_concrete['age']):.2f} MPa")
    print(f"Gradient Boosting: {predict_strength(gb_model, scaler, new_concrete['cement'], new_concrete['pofa'], new_concrete['ca_fa'], new_concrete['w_b'], new_concrete['sp'], new_concrete['age']):.2f} MPa")
    print(f"SVR: {predict_strength(svr_model, scaler, new_concrete['cement'], new_concrete['pofa'], new_concrete['ca_fa'], new_concrete['w_b'], new_concrete['sp'], new_concrete['age']):.2f} MPa")
    print(f"Tuned Random Forest: {predict_strength(tuned_rf_model, scaler, new_concrete['cement'], new_concrete['pofa'], new_concrete['ca_fa'], new_concrete['w_b'], new_concrete['sp'], new_concrete['age']):.2f} MPa")
    print(f"Hybrid Model: {predict_strength(hybrid_model_trained, scaler, new_concrete['cement'], new_concrete['pofa'], new_concrete['ca_fa'], new_concrete['w_b'], new_concrete['sp'], new_concrete['age']):.2f} MPa")

if __name__ == "__main__":
    compare_all_models()