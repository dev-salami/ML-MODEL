"""
M5P Model Tree implementation for POFA concrete strength prediction.
This module implements M5P model trees which combine decision trees with linear regression models at the leaves.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import warnings

# Filter warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

class M5PModel(BaseEstimator, RegressorMixin):
    """
    M5P Model Tree implementation - Decision tree with linear regression models at the leaves.
    
    Inspired by Quinlan's M5 algorithm and Wang & Witten's M5' improvements.
    Implements model trees where each leaf contains a linear regression model.
    
    Parameters:
    -----------
    max_depth : int, default=5
        Maximum depth of the decision tree
    min_samples_split : int, default=10
        Minimum number of samples required to split an internal node
    min_samples_leaf : int, default=5
        Minimum number of samples required to be at a leaf node
    """
    
    def __init__(self, max_depth=5, min_samples_split=10, min_samples_leaf=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        self.leaf_models = {}
        self.feature_names = None
        
    def fit(self, X, y):
        """
        Fit the M5P model tree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training input samples
        y : array-like of shape (n_samples,)
            Target values
            
        Returns:
        --------
        self : object
            Returns self
        """
        # Store feature names if available
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
        
        # Convert to numpy arrays if needed
        X = np.asarray(X)
        y = np.asarray(y)
        
        # First fit a decision tree to partition the space
        self.tree = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=42
        )
        self.tree.fit(X, y)
        
        # Get the leaf node assignments for each training instance
        leaf_ids = self.tree.apply(X)
        
        # Fit a linear regression model for each leaf
        for leaf_id in np.unique(leaf_ids):
            # Get the indices of samples in this leaf
            indices = np.where(leaf_ids == leaf_id)[0]
            
            # Skip if too few samples
            if len(indices) <= self.min_samples_leaf:
                continue
                
            # Fit a linear regression model
            leaf_model = LinearRegression()
            leaf_model.fit(X[indices], y[indices])
            
            # Store the leaf model
            self.leaf_models[leaf_id] = leaf_model
            
        return self
    
    def predict(self, X):
        """
        Predict using the M5P model tree.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Input samples
            
        Returns:
        --------
        y_pred : array-like of shape (n_samples,)
            Predicted values
        """
        # Convert to numpy array if needed
        X = np.asarray(X)
        
        # Get the leaf node assignments for each instance
        leaf_ids = self.tree.apply(X)
        
        # Initialize predictions array
        y_pred = np.zeros(X.shape[0])
        
        # Predict using the appropriate linear model for each leaf
        for i, leaf_id in enumerate(leaf_ids):
            if leaf_id in self.leaf_models:
                # Use the leaf's linear model
                y_pred[i] = self.leaf_models[leaf_id].predict([X[i]])[0]
            else:
                # Fall back to the decision tree's prediction if no model exists for this leaf
                # This might happen if a leaf had too few samples during training
                y_pred[i] = self.tree.predict([X[i]])[0]
                
        return y_pred
    
    def score(self, X, y):
        """
        Return the R^2 score of the model on the given test data.
        
        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Test samples
        y : array-like of shape (n_samples,)
            True values for X
            
        Returns:
        --------
        score : float
            R^2 score
        """
        return r2_score(y, self.predict(X))
    
    def get_feature_importance(self):
        """
        Get feature importances from the underlying decision tree.
        
        Returns:
        --------
        feature_importances : array of shape (n_features,)
            Feature importances
        """
        return self.tree.feature_importances_


def train_m5p_model(X_train, y_train, X_test, y_test):
    """
    Train an M5P model tree with default parameters.
    
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
        
    Returns:
    --------
    model : M5PModel
        Trained M5P model
    metrics : dict
        Performance metrics
    """
    # Create and train the model
    model = M5PModel()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print performance metrics
    print("\nM5P Model Evaluation:")
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
    
    plt.title('M5P Model: Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('m5p_actual_vs_predicted.png')
    plt.close()
    
    # Return model and metrics
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    return model, metrics


def tune_m5p_model(X_train, y_train, X_test, y_test):
    """
    Tune the M5P model using grid search cross-validation.
    
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
        
    Returns:
    --------
    model : M5PModel
        Tuned M5P model
    metrics : dict
        Performance metrics
    """
    print("\nTuning M5P model hyperparameters...")
    
    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [5, 10, 15, 20],
        'min_samples_leaf': [3, 5, 10]
    }
    
    # Create grid search
    model = M5PModel()
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
    tuned_model = M5PModel(
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf']
    )
    tuned_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = tuned_model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Print performance metrics
    print("\nTuned M5P Model Evaluation:")
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
    
    plt.title('Tuned M5P Model: Actual vs Predicted Compressive Strength')
    plt.xlabel('Actual Strength (MPa)')
    plt.ylabel('Predicted Strength (MPa)')
    plt.grid(True, alpha=0.3)
    
    # Add R² annotation
    plt.annotate(f'R² = {r2:.4f}', xy=(0.05, 0.95), xycoords='axes fraction',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('tuned_m5p_actual_vs_predicted.png')
    plt.close()
    
    # Return tuned model and metrics
    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}
    return tuned_model, metrics


def feature_importance_m5p(model, feature_names=None):
    """
    Plot feature importance for the M5P model.
    
    Parameters:
    -----------
    model : M5PModel
        Trained M5P model
    feature_names : list, optional
        Names of features, if None will use indices
    """
    # Get feature importances
    importances = model.get_feature_importance()
    
    # Use feature names if provided, otherwise use indices
    if feature_names is None:
        if model.feature_names is not None:
            feature_names = model.feature_names
        else:
            feature_names = [f'Feature {i}' for i in range(len(importances))]
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_feature_names = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), sorted_importances, align='center')
    plt.xticks(range(len(importances)), sorted_feature_names, rotation=45, ha='right')
    plt.title('M5P Model Feature Importance')
    plt.tight_layout()
    plt.savefig('m5p_feature_importance.png')
    plt.close()


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    
    # Generate sample data
    X, y = make_regression(n_samples=1000, n_features=6, noise=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train and evaluate M5P model
    model, metrics = train_m5p_model(X_train, y_train, X_test, y_test)
    
    # Tune M5P model
    tuned_model, tuned_metrics = tune_m5p_model(X_train, y_train, X_test, y_test)
    
    # Plot feature importance
    feature_importance_m5p(tuned_model)