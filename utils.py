import os
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

BASE_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', 'v1')

OUTPUT_DIRS = {
    'actual_vs_predicted': os.path.join(BASE_OUT, 'actual_vs_predicted'),
    'shap':                os.path.join(BASE_OUT, 'shap'),
    'pdp':                 os.path.join(BASE_OUT, 'pdp'),
    'feature_importance':  os.path.join(BASE_OUT, 'feature_importance'),
    'correlation':         os.path.join(BASE_OUT, 'correlation'),
    'training_loss':       os.path.join(BASE_OUT, 'training_loss'),
    'misc':                os.path.join(BASE_OUT, 'misc'),
    'optuna':              os.path.join(BASE_OUT, 'optuna'),
    'models':              os.path.join(BASE_OUT, 'models'),
}

def ensure_output_dirs():
    for path in OUTPUT_DIRS.values():
        os.makedirs(path, exist_ok=True)

def get_output_dirs(version='v1'):
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'out', version)
    return {
        'actual_vs_predicted': os.path.join(base, 'actual_vs_predicted'),
        'shap':                os.path.join(base, 'shap'),
        'pdp':                 os.path.join(base, 'pdp'),
        'feature_importance':  os.path.join(base, 'feature_importance'),
        'correlation':         os.path.join(base, 'correlation'),
        'training_loss':       os.path.join(base, 'training_loss'),
        'misc':                os.path.join(base, 'misc'),
        'optuna':              os.path.join(base, 'optuna'),
        'models':              os.path.join(base, 'models'),
    }

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """
    Train and evaluate a model's performance.
    
    Parameters:
    -----------
    model: ML model object
    X_train: Training features
    X_test: Test features
    y_train: Training target values
    y_test: Test target values
    model_name: Name of the model (string)
    
    Returns:
    --------
    model: Trained model
    y_pred: Predicted values on test set
    mse: Mean Squared Error
    r2: R-squared score
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print("-" * 50)
    
    return model, y_pred, mse, r2

def predict_strength(model, scaler, cement, pofa, ca_fa, w_b, sp, age):
    """
    Predict concrete compressive strength using the trained model.
    
    Parameters:
    -----------
    model: The trained ML model
    scaler: The feature scaler
    cement: Cement content (kg/m³)
    pofa: Palm Oil Fuel Ash content (kg/m³)
    ca_fa: Coarse Aggregate to Fine Aggregate ratio
    w_b: Water to Binder ratio
    sp: Superplasticizer content (kg/m³)
    age: Age of concrete (days)
    
    Returns:
    --------
    float: Predicted compressive strength (MPa)
    """
    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None

    features = [cement, pofa, ca_fa, w_b, sp, age]

    if n_features == 7:
        # Legacy scalers were fit with the DataPoint id as the leading column
        # ([DataPoint, C, POFA, CA/FA, W/B, SP, Age]). Prepend a neutral id so the
        # six real inputs still line up with the columns the scaler expects.
        input_data = np.array([[201] + features])
    else:
        input_data = np.array([features])
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    return prediction[0]

def plot_model_comparison(model_results):
    """
    Create a bar chart comparing the R² scores of different models.
    
    Parameters:
    -----------
    model_results: Dictionary containing model names and their R² scores
    """
    plt.figure(figsize=(10, 6))
    models = list(model_results.keys())
    r2_scores = [model_results[model] for model in models]
    
    plt.bar(models, r2_scores)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Comparison - R² Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS['misc'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['misc'], 'model_comparison.png'))
    plt.close()
    
def save_model(model, filename):
    """
    Save a trained model to disk.

    Parameters:
    -----------
    model: Trained model to save
    filename: Name of the file to save the model to
    """
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved to {filename}")

def load_model(filename):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    filename: Name of the file containing the saved model
    
    Returns:
    --------
    model: Loaded model
    """
    return joblib.load(filename)