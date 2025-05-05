import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import joblib

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
    # Check how many features the scaler expects
    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else None
    
    if n_features and n_features == 7:
        # The scaler was trained with 7 features (including concrete_strength as a dummy)
        # Add a dummy value (0) for the concrete_strength parameter
        input_data = np.array([[cement, pofa, ca_fa, w_b, sp, age, 25]])
    else:
        # Standard case with 6 features
        input_data = np.array([[cement, pofa, ca_fa, w_b, sp, age]])
    
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
    plt.savefig('model_comparison.png')
    plt.close()
    
def save_model(model, filename):
    """
    Save a trained model to disk.
    
    Parameters:
    -----------
    model: Trained model to save
    filename: Name of the file to save the model to
    """
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