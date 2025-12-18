"""
Artificial Neural Network (ANN) model for POFA concrete strength prediction using scikit-learn.
This is a fallback implementation that doesn't require TensorFlow.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import joblib

def create_ann_sklearn(random_state=42):
    """
    Create the ANN model using scikit-learn's MLPRegressor.
    
    Args:
        random_state: Random state for reproducibility
        
    Returns:
        MLPRegressor model
    """
    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=random_state,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        verbose=False
    )
    
    return model

def train_ann_sklearn(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Train the ANN model using scikit-learn.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target values
        X_test_scaled: Scaled test features
        y_test: Test target values
        
    Returns:
        Trained model and evaluation results
    """
    # Create and train model
    model = create_ann_sklearn()
    
    # Train the model
    print("Training ANN model (scikit-learn)...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print results
    print("\nANN Model (scikit-learn) Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Print training information
    print(f"Number of iterations: {model.n_iter_}")
    print(f"Number of layers: {model.n_layers_}")
    print(f"Number of outputs: {model.n_outputs_}")
    
    # Plot training loss curve
    if hasattr(model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(model.loss_curve_)
        plt.title('ANN Training Loss Curve (scikit-learn)')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('ann_sklearn_training_loss.png')
        plt.close()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('ANN Model (scikit-learn): Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('ann_sklearn_actual_vs_predicted.png')
    plt.close()
    
    return model, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def tune_ann_sklearn(X_train_scaled, y_train, X_test_scaled, y_test):
    """
    Tune the ANN model hyperparameters using scikit-learn.
    
    Args:
        X_train_scaled: Scaled training features
        y_train: Training target values
        X_test_scaled: Scaled test features
        y_test: Test target values
        
    Returns:
        Best model and evaluation results
    """
    # Define parameter grid
    param_grid = {
        'hidden_layer_sizes': [(64, 32), (64, 32, 16), (128, 64, 32), (100, 50), (200, 100, 50)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1],
        'max_iter': [300, 500, 800]
    }
    
    # Create base model
    base_model = MLPRegressor(
        activation='relu',
        solver='adam',
        batch_size='auto',
        learning_rate='adaptive',
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=30,
        verbose=False
    )
    
    # Perform Grid Search
    print("Tuning ANN hyperparameters (this may take a while)...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1
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
    print("\nBest ANN Parameters (scikit-learn):")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    # Print results
    print("\nTuned ANN Model (scikit-learn) Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Print training information
    print(f"Number of iterations: {best_model.n_iter_}")
    print(f"Number of layers: {best_model.n_layers_}")
    
    # Plot training loss curve
    if hasattr(best_model, 'loss_curve_'):
        plt.figure(figsize=(10, 6))
        plt.plot(best_model.loss_curve_)
        plt.title('Tuned ANN Training Loss Curve (scikit-learn)')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('tuned_ann_sklearn_training_loss.png')
        plt.close()
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Tuned ANN Model (scikit-learn): Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('tuned_ann_sklearn_actual_vs_predicted.png')
    plt.close()
    
    return best_model, {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}