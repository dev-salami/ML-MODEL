"""
MARS (Multivariate Adaptive Regression Splines) model for POFA concrete strength prediction.
This module implements MARS which automatically models non-linearities and interactions between variables.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings
from utils import OUTPUT_DIRS

# Filter warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

def train_mars_model(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Train a MARS model with default parameters.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training input samples
    y_train : array-like of shape (n_samples,)
        Target values for training
    X_test : array-like of shape (n_samples, n_features)
        Test input samples
    y_test : array-like of shape (n_samples,)
        Target values for testing
    feature_names : list, optional
        Names of features
        
    Returns:
    --------
    model : Earth (py-earth)
        Trained MARS model
    metrics : dict
        Performance metrics
    """
    try:
        from pyearth import Earth
    except ImportError:
        print("The py-earth package is required for MARS modeling.")
        print("Please install it using: pip install pyearth")
        return None, None
    
    # Create and train the model
    model = Earth(max_degree=2,  # Max degree of interactions
                 penalty=3.0,    # GCV penalty
                 minspan=2,      # Minimum number of observations between knots
                 endspan=10,     # Minimum number of observations before the end of the data
                 allow_linear=True,
                 enable_pruning=True,
                 verbose=1)
    
    model.fit(X_train, y_train)
    
    # Print model summary
    print("\nMARS Model Summary:")
    print(model.summary())
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print performance metrics
    print("\nMARS Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create actual vs. predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('MARS Model: Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS['actual_vs_predicted'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['actual_vs_predicted'], 'mars_actual_vs_predicted.png'))
    plt.close()
    
    # Plot coefficients
    try:
        # Get the basis functions
        basis_functions = model.transform(X_test)
        
        # Plot coefficient values
        coef = model.coef_
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(coef)), coef)
        plt.title('MARS Model Coefficients')
        plt.xlabel('Basis Function')
        plt.ylabel('Coefficient Value')
        plt.tight_layout()
        os.makedirs(OUTPUT_DIRS['misc'], exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIRS['misc'], 'mars_coefficients.png'))
        plt.close()
        
        # Plot feature importance if available
        plot_mars_feature_importance(model, feature_names)
        
    except Exception as e:
        print(f"Warning: Could not generate MARS coefficient plots: {e}")
    
    # Return model and metrics
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    return model, metrics


def tune_mars_model(X_train, y_train, X_test, y_test, feature_names=None):
    """
    Tune the MARS model using grid search cross-validation.
    
    Parameters:
    -----------
    X_train : array-like of shape (n_samples, n_features)
        Training input samples
    y_train : array-like of shape (n_samples,)
        Target values for training
    X_test : array-like of shape (n_samples, n_features)
        Test input samples
    y_test : array-like of shape (n_samples,)
        Target values for testing
    feature_names : list, optional
        Names of features
        
    Returns:
    --------
    model : Earth (py-earth)
        Tuned MARS model
    metrics : dict
        Performance metrics
    """
    try:
        from pyearth import Earth
    except ImportError:
        print("The py-earth package is required for MARS modeling.")
        print("Please install it using: pip install pyearth")
        return None, None
    
    print("\nTuning MARS model hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'max_degree': [1, 2, 3],
        'penalty': [2, 3, 4],
        'min_search_points': [10, 20, 30],
        'allow_linear': [True, False],
        'enable_pruning': [True]
    }
    
    # Create grid search
    model = Earth()
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=0
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    print("Best parameters:", best_params)
    
    # Create and train the model with best parameters
    tuned_model = Earth(
        max_degree=best_params['max_degree'],
        penalty=best_params['penalty'],
        min_search_points=best_params['min_search_points'],
        allow_linear=best_params['allow_linear'],
        enable_pruning=best_params['enable_pruning'],
        verbose=1
    )
    tuned_model.fit(X_train, y_train)
    
    # Print model summary
    print("\nTuned MARS Model Summary:")
    print(tuned_model.summary())
    
    # Make predictions
    y_pred = tuned_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print performance metrics
    print("\nTuned MARS Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create actual vs. predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(y_pred))
    max_val = max(max(y_test), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    
    plt.title('Tuned MARS Model: Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS['actual_vs_predicted'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['actual_vs_predicted'], 'tuned_mars_actual_vs_predicted.png'))
    plt.close()
    
    # Plot feature importance if available
    plot_mars_feature_importance(tuned_model, feature_names)
    
    # Return tuned model and metrics
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    return tuned_model, metrics


def plot_mars_feature_importance(model, feature_names=None):
    """
    Plot feature importance for the MARS model.
    
    Parameters:
    -----------
    model : Earth (py-earth)
        Trained MARS model
    feature_names : list, optional
        Names of features, if None will use indices
    """
    try:
        # Get feature importance
        importances = model.summary_feature_importances()
        
        # Use feature names if provided, otherwise use indices
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
        
        # Ensure feature_names has the right length
        if len(feature_names) > len(importances):
            feature_names = feature_names[:len(importances)]
        elif len(feature_names) < len(importances):
            feature_names = feature_names + [f'Feature {i}' for i in range(len(feature_names), len(importances))]
        
        # Sort features by importance
        sorted_indices = np.argsort(importances)[::-1]
        sorted_importances = np.array(importances)[sorted_indices]
        sorted_feature_names = np.array(feature_names)[sorted_indices]
        
        # Plot feature importances
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(sorted_importances)), sorted_importances)
        plt.xticks(range(len(sorted_importances)), sorted_feature_names, rotation=45, ha='right')
        plt.title('MARS Model Feature Importance')
        plt.tight_layout()
        os.makedirs(OUTPUT_DIRS['feature_importance'], exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIRS['feature_importance'], 'mars_feature_importance.png'))
        plt.close()
        
    except Exception as e:
        print(f"Warning: Could not generate MARS feature importance plot: {e}")


def plot_mars_hinge_functions(model, X, feature_names=None, num_points=1000):
    """
    Plot selected hinge functions from the MARS model.
    
    Parameters:
    -----------
    model : Earth (py-earth)
        Trained MARS model
    X : array-like of shape (n_samples, n_features)
        Input data to find the range of feature values
    feature_names : list, optional
        Names of features
    num_points : int, default=1000
        Number of points to use for plotting hinge functions
    """
    try:
        from pyearth._basis import ConstantBasisFunction, HingeBasisFunction, LinearBasisFunction
        
        # Get basis functions
        basis_functions = model.basis_
        
        # Use feature names if provided, otherwise use indices
        if feature_names is None:
            feature_names = [f'Feature {i}' for i in range(X.shape[1])]
        
        # Convert X to numpy array if needed
        X = np.asarray(X)
        
        # Plot selected hinge functions (up to 6)
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        # Find hinge functions to plot
        hinge_funcs = []
        for bf in basis_functions:
            if isinstance(bf, HingeBasisFunction) or isinstance(bf, LinearBasisFunction):
                if len(hinge_funcs) < 6:  # Limit to 6 for visualization
                    hinge_funcs.append(bf)
        
        # Plot each selected function
        for i, bf in enumerate(hinge_funcs):
            if i >= 6:  # Only plot up to 6 functions
                break
                
            # Get the feature index for this basis function
            if hasattr(bf, 'get_variable'):
                feature_idx = bf.get_variable()
                feature_name = feature_names[feature_idx]
            else:
                continue
            
            # Get the range of values for this feature
            feature_min = np.min(X[:, feature_idx])
            feature_max = np.max(X[:, feature_idx])
            x_values = np.linspace(feature_min, feature_max, num_points)
            
            # Create a dataset with just this feature varying
            X_temp = np.zeros((num_points, X.shape[1]))
            X_mean = np.mean(X, axis=0)
            X_temp[:] = X_mean  # Set all features to their mean value
            X_temp[:, feature_idx] = x_values  # Vary only the target feature
            
            # Get the model's prediction for these data points
            y_pred = model.predict(X_temp)
            
            # Plot the hinge function
            axes[i].plot(x_values, y_pred)
            axes[i].set_title(f'Effect of {feature_name}')
            axes[i].set_xlabel(feature_name)
            axes[i].set_ylabel('Predicted Value')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(OUTPUT_DIRS['misc'], exist_ok=True)
        plt.savefig(os.path.join(OUTPUT_DIRS['misc'], 'mars_hinge_functions.png'))