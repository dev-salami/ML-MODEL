"""
Bagging algorithm implementation for POFA concrete strength prediction.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib
import sklearn

def create_bagging_regressor(base_estimator, **kwargs):
    """
    Create BaggingRegressor with compatibility for different scikit-learn versions.
    """
    # Check scikit-learn version and use appropriate parameter name
    sklearn_version = sklearn.__version__
    major_version = int(sklearn_version.split('.')[0])
    minor_version = int(sklearn_version.split('.')[1])
    
    # In scikit-learn 1.2+, base_estimator was renamed to estimator
    if major_version > 1 or (major_version == 1 and minor_version >= 2):
        return BaggingRegressor(estimator=base_estimator, **kwargs)
    else:
        return BaggingRegressor(base_estimator=base_estimator, **kwargs)

def train_bagging(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Train a Bagging Regressor model.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target values
        X_test_scaled: Scaled test features
        y_test: Test target values
        
    Returns:
        Trained model and evaluation results
    """
    # Create base estimator (Decision Tree Regressor)
    base_estimator = DecisionTreeRegressor(random_state=42)
    
    # Create and train the Bagging model
    bagging_model = create_bagging_regressor(
        base_estimator,
        n_estimators=100,
        max_samples=0.8,
        max_features=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42
    )
    
    # Train the model
    bagging_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = bagging_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print("\nBagging Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Bagging Model: Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('bagging_actual_vs_predicted.png')
    plt.close()
    
    return bagging_model, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def tune_bagging(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Tune the Bagging Regressor model hyperparameters.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target values
        X_test_scaled: Scaled test features
        y_test: Test target values
        
    Returns:
        Best model and evaluation results
    """
    # Create base estimator
    base_estimator = DecisionTreeRegressor(random_state=42)
    
    # Create Bagging model
    bagging_model = create_bagging_regressor(
        base_estimator,
        random_state=42
    )
    
    # Define parameter grid for hyperparameter tuning (simplified)
    param_grid = {
        'n_estimators': [50, 100],
        'max_samples': [0.7, 0.9],
        'max_features': [0.7, 0.9],
        'bootstrap': [True]
    }
    
    # Perform Grid Search
    grid_search = GridSearchCV(
        estimator=bagging_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=1  # Changed from -1 to 1 to avoid multiprocessing issues
    )
    
    # Fit the grid search
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model
    best_model = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print best parameters
    print("\nBest Bagging Model Parameters:")
    print(grid_search.best_params_)
    
    # Print results
    print("\nTuned Bagging Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Tuned Bagging Model: Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('tuned_bagging_actual_vs_predicted.png')
    plt.close()
    
    return best_model, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def feature_importance_bagging(model, feature_names):
    """
    Extract and visualize feature importances from the Bagging model.
    
    Args:
        model: Trained Bagging model
        feature_names: Names of the input features
        
    Returns:
        None (creates a visualization)
    """
    try:
        # For Bagging with DecisionTreeRegressor, we can average feature importances
        # across all base estimators
        importances = np.zeros(len(feature_names))
        valid_estimators = 0
        
        for estimator in model.estimators_:
            # Check if the estimator has the expected number of features
            if hasattr(estimator, 'feature_importances_') and len(estimator.feature_importances_) == len(feature_names):
                importances += estimator.feature_importances_
                valid_estimators += 1
            elif hasattr(estimator, 'feature_importances_'):
                # Handle case where estimator has different number of features
                min_features = min(len(estimator.feature_importances_), len(feature_names))
                importances[:min_features] += estimator.feature_importances_[:min_features]
                valid_estimators += 1
        
        if valid_estimators > 0:
            # Average importances
            importances /= valid_estimators
            
            # Sort by importance
            indices = np.argsort(importances)
            
            # Plot feature importances
            plt.figure(figsize=(10, 6))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Average Feature Importance')
            plt.title('Bagging Model: Feature Importance')
            plt.tight_layout()
            plt.savefig('bagging_feature_importance.png')
            plt.close()
            
            print("\nBagging Model Feature Importances:")
            for i in indices[::-1]:  # Print in descending order
                print(f"{feature_names[i]}: {importances[i]:.4f}")
        else:
            print("Could not extract feature importances from bagging model.")
            
    except Exception as e:
        print(f"Error calculating feature importances: {e}")
        print("Skipping feature importance visualization.")