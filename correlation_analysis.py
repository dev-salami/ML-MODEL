"""
Pearson correlation analysis for predicted vs actual concrete strength values.
This script generates correlation plots for all models.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import joblib
import os

def plot_correlation_matrix(models_predictions, actual, model_names):
    """
    Generate a correlation matrix plot for all models' predictions vs actual values.
    
    Args:
        models_predictions: Dictionary of model predictions
        actual: Actual target values
        model_names: List of model names
        
    Returns:
        None (saves the plot)
    """
    # Create DataFrame with actual and all predictions
    data = {'Actual': actual}
    
    for name, pred in models_predictions.items():
        data[f'{name}_Predicted'] = pred
    
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    corr_matrix = df.corr(method='pearson')
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    heatmap = sns.heatmap(
        corr_matrix, 
        annot=True, 
        mask=mask, 
        cmap=cmap, 
        vmin=-1, 
        vmax=1, 
        center=0,
        fmt='.3f',
        square=True, 
        linewidths=.5
    )
    
    plt.title('Pearson Correlation Matrix: Actual vs Predicted Values', fontsize=16)
    plt.tight_layout()
    plt.savefig('correlation_matrix.png', dpi=300)
    plt.close()
    
    return corr_matrix

def plot_individual_correlations(models_predictions, actual, model_names):
    """
    Generate individual correlation scatter plots for each model.
    
    Args:
        models_predictions: Dictionary of model predictions
        actual: Actual target values
        model_names: List of model names
        
    Returns:
        Dictionary of correlation coefficients and p-values
    """
    correlation_results = {}
    
    # Create a figure with subplots for each model
    n_models = len(model_names)
    n_cols = 2
    n_rows = (n_models + 1) // 2  # Ceiling division
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i, name in enumerate(model_names):
        predictions = models_predictions[name]
        
        # Calculate Pearson correlation
        corr, p_value = pearsonr(actual, predictions)
        correlation_results[name] = {'correlation': corr, 'p_value': p_value}
        
        # Create subplot
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Plot scatter with regression line
        sns.regplot(x=actual, y=predictions, scatter_kws={'alpha': 0.7}, line_kws={'color': 'red'})
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predictions))
        max_val = max(max(actual), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1)
        
        plt.title(f'{name} Model: Actual vs Predicted Strength', fontsize=14)
        plt.xlabel('Actual Strength (MPa)', fontsize=12)
        plt.ylabel('Predicted Strength (MPa)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add correlation annotation
        plt.annotate(
            f'Pearson r = {corr:.4f}\np-value = {p_value:.4e}', 
            xy=(0.05, 0.92), 
            xycoords='axes fraction',
            bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
        )
    
    plt.tight_layout()
    plt.savefig('individual_correlations.png', dpi=300)
    plt.close()
    
    return correlation_results

def pearson_correlation_table(correlation_results, model_names):
    """
    Create a formatted table of correlation results.
    
    Args:
        correlation_results: Dictionary of correlation results
        model_names: List of model names
        
    Returns:
        None (prints table)
    """
    # Print table header
    print("\nPearson Correlation Results:")
    print("-" * 70)
    print(f"{'Model':<20} {'Correlation (r)':<20} {'p-value':<15} {'Significance':<15}")
    print("-" * 70)
    
    # Sort models by correlation (descending)
    sorted_models = sorted(
        model_names, 
        key=lambda x: correlation_results[x]['correlation'], 
        reverse=True
    )
    
    # Print each model's results
    for name in sorted_models:
        corr = correlation_results[name]['correlation']
        p_value = correlation_results[name]['p_value']
        
        # Determine significance
        if p_value < 0.001:
            significance = "p < 0.001 ***"
        elif p_value < 0.01:
            significance = "p < 0.01 **"
        elif p_value < 0.05:
            significance = "p < 0.05 *"
        else:
            significance = "Not significant"
        
        print(f"{name:<20} {corr:<20.4f} {p_value:<15.4e} {significance:<15}")
    
    print("-" * 70)

def generate_correlation_analysis(X_test, y_test, scaler, feature_names):
    """
    Generate correlation analysis for all available models.
    
    Args:
        X_test: Test features
        y_test: Test target values
        scaler: Feature scaler
        feature_names: Names of features
        
    Returns:
        None (generates plots and reports)
    """
    # Dictionary to store model predictions
    models_predictions = {}
    model_names = []
    
    # List of all possible model files
    model_files = [
        ('Linear Regression', 'linear_regression_model.pkl'),
        ('Random Forest', 'random_forest_model_tuned.pkl'),
        ('Gradient Boosting', 'gradient_boosting_model_tuned.pkl'),
        ('SVR', 'svr_model_tuned.pkl'),
        ('Hybrid', 'hybrid_model.pkl'),
        ('Bagging', 'bagging_model_tuned.pkl'),
        ('GEP', 'gep_model.pkl')
    ]
    
    # Load each available model and get predictions
    for name, filename in model_files:
        if os.path.exists(filename):
            print(f"Loading {name} model...")
            
            if name == 'GEP':
                try:
                    gep_data = joblib.load(filename)
                    gep_func = gep_data['function']
                    
                    # Get predictions using GEP function
                    predictions = np.zeros(len(y_test))
                    for i in range(len(X_test)):
                        try:
                            predictions[i] = gep_func(*X_test[i])
                            if np.isnan(predictions[i]) or np.isinf(predictions[i]):
                                predictions[i] = np.mean(y_test)
                        except Exception:
                            predictions[i] = np.mean(y_test)
                            
                    models_predictions[name] = predictions
                    model_names.append(name)
                except Exception as e:
                    print(f"Error loading GEP model: {e}")
            else:
                try:
                    model = joblib.load(filename)
                    predictions = model.predict(X_test)
                    models_predictions[name] = predictions
                    model_names.append(name)
                except Exception as e:
                    print(f"Error loading {name} model: {e}")
    
    if not model_names:
        print("No models found. Please train models first.")
        return
    
    print(f"Loaded {len(model_names)} models for correlation analysis.")
    
    # Generate correlation matrix
    corr_matrix = plot_correlation_matrix(models_predictions, y_test, model_names)
    
    # Generate individual correlation plots
    correlation_results = plot_individual_correlations(models_predictions, y_test, model_names)
    
    # Print correlation table
    pearson_correlation_table(correlation_results, model_names)
    
    # Create detailed scatter plot for the best model
    best_model = max(correlation_results.items(), key=lambda x: x[1]['correlation'])[0]
    
    plt.figure(figsize=(10, 8))
    
    # Plot with more details
    sns.regplot(
        x=y_test, 
        y=models_predictions[best_model],
        scatter_kws={'alpha': 0.7, 's': 80, 'edgecolor': 'k'},
        line_kws={'color': 'red', 'lw': 2}
    )
    
    # Add perfect prediction line
    min_val = min(min(y_test), min(models_predictions[best_model]))
    max_val = max(max(y_test), max(models_predictions[best_model]))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    
    # Calculate confidence intervals
    residuals = models_predictions[best_model] - y_test
    rmse = np.sqrt(np.mean(residuals**2))
    
    # Add confidence interval lines (±RMSE)
    x_range = np.linspace(min_val, max_val, 100)
    plt.fill_between(
        x_range, 
        x_range - rmse, 
        x_range + rmse,
        alpha=0.2, 
        color='gray', 
        label=f'±RMSE ({rmse:.2f} MPa)'
    )
    
    plt.title(f'Best Model ({best_model}): Pearson Correlation Analysis', fontsize=16)
    plt.xlabel('Actual Strength (MPa)', fontsize=14)
    plt.ylabel('Predicted Strength (MPa)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add correlation annotation
    corr = correlation_results[best_model]['correlation']
    p_value = correlation_results[best_model]['p_value']
    
    plt.annotate(
        f'Pearson r = {corr:.4f}\np-value = {p_value:.4e}', 
        xy=(0.05, 0.92), 
        xycoords='axes fraction',
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
        fontsize=12
    )
    
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('best_model_correlation.png', dpi=300)
    plt.close()
    
    print(f"\nBest performing model based on Pearson correlation: {best_model} (r = {corr:.4f})")
    print("Correlation analysis complete. Saved plots: correlation_matrix.png, individual_correlations.png, best_model_correlation.png")