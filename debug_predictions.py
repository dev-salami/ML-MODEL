import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

from utils import predict_strength

def verify_predictions(model_file='random_forest_model_tuned.pkl', 
                     scaler_file='pofa_concrete_strength_scaler.pkl',
                     data_file='pofa_concrete_data.csv',
                     sample_count=10):
    """
    Verify model predictions against actual values from the dataset.
    
    Parameters:
    -----------
    model_file: Path to the trained model file
    scaler_file: Path to the scaler file
    data_file: Path to the CSV data file
    sample_count: Number of samples to check
    """
    try:
        # Load model and scaler
        model = joblib.load(model_file)
        scaler = joblib.load(scaler_file)
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Randomly select samples for verification
        samples = df.sample(sample_count, random_state=42)
        
        # Prepare results
        results = []
        
        # For each sample, compare actual vs predicted
        for _, row in samples.iterrows():
            # Extract features
            cement = row['C']
            pofa = row['POFA']
            ca_fa = row['CA/FA']
            w_b = row['W/B']
            sp = row['SP']
            age = row['Age']
            actual_cs = row['CS']
            
            # Make prediction
            predicted_cs = predict_strength(
                model, scaler, cement, pofa, ca_fa, w_b, sp, age
            )
            
            # Calculate error and percentage error
            error = actual_cs - predicted_cs
            percent_error = (error / actual_cs) * 100
            
            # Store results
            results.append({
                'Cement': cement,
                'POFA': pofa,
                'CA/FA': ca_fa,
                'W/B': w_b,
                'SP': sp,
                'Age': age,
                'Actual CS': actual_cs,
                'Predicted CS': predicted_cs,
                'Error': error,
                'Error %': percent_error
            })
        
        # Convert to DataFrame for better display
        results_df = pd.DataFrame(results)
        
        # Calculate overall metrics
        mse = mean_squared_error(results_df['Actual CS'], results_df['Predicted CS'])
        rmse = np.sqrt(mse)
        r2 = r2_score(results_df['Actual CS'], results_df['Predicted CS'])
        mean_abs_error = np.abs(results_df['Error']).mean()
        mean_pct_error = np.abs(results_df['Error %']).mean()
        
        # Print summary
        print("\n----- Model Prediction Verification -----")
        print(f"Model: {model_file}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²: {r2:.4f}")
        print(f"Mean Absolute Error: {mean_abs_error:.2f}")
        print(f"Mean Absolute Percentage Error: {mean_pct_error:.2f}%")
        
        # Print sample comparison
        print("\n----- Sample Results -----")
        print(results_df[['Actual CS', 'Predicted CS', 'Error', 'Error %']].to_string(index=False))
        
        # Visualize
        plt.figure(figsize=(10, 6))
        plt.scatter(results_df['Actual CS'], results_df['Predicted CS'])
        plt.plot([min(results_df['Actual CS']), max(results_df['Actual CS'])], 
                [min(results_df['Actual CS']), max(results_df['Actual CS'])], 
                'r--')
        plt.xlabel('Actual Compressive Strength (MPa)')
        plt.ylabel('Predicted Compressive Strength (MPa)')
        plt.title('Actual vs Predicted Compressive Strength')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('prediction_verification.png')
        plt.close()
        
        # Return results for further analysis
        return results_df, model, scaler
        
    except Exception as e:
        print(f"Error: {e}")
        return None, None, None

def find_similar_samples(sample_params, data_file='pofa_concrete_data.csv', top_n=5):
    """
    Find the most similar concrete samples in the dataset to compare with predictions.
    
    Parameters:
    -----------
    sample_params: Dictionary of sample parameters (cement, pofa, ca_fa, w_b, sp, age)
    data_file: Path to the CSV data file
    top_n: Number of similar samples to find
    """
    try:
        # Load data
        df = pd.read_csv(data_file)
        
        # Calculate similarity score (Euclidean distance with simple normalization)
        # First normalize the values to 0-1 range to give equal weight
        df_norm = df.copy()
        for col in ['C', 'POFA', 'CA/FA', 'W/B', 'SP', 'Age']:
            col_min = df[col].min()
            col_max = df[col].max()
            df_norm[col] = (df[col] - col_min) / (col_max - col_min) if col_max > col_min else 0
        
        # Normalize sample parameters
        sample_norm = {}
        for col in ['C', 'POFA', 'CA/FA', 'W/B', 'SP', 'Age']:
            col_key = col
            if col == 'C': col_key = 'cement'
            elif col == 'POFA': col_key = 'pofa'
            elif col == 'CA/FA': col_key = 'ca_fa'
            elif col == 'W/B': col_key = 'w_b'
            elif col == 'SP': col_key = 'sp'
            elif col == 'Age': col_key = 'age'
            
            col_min = df[col].min()
            col_max = df[col].max()
            if col_key in sample_params:
                sample_norm[col] = (sample_params[col_key] - col_min) / (col_max - col_min) if col_max > col_min else 0
        
        # Calculate distance for each sample
        distances = []
        for i, row in df_norm.iterrows():
            distance = np.sqrt(
                (row['C'] - sample_norm['C'])**2 +
                (row['POFA'] - sample_norm['POFA'])**2 +
                (row['CA/FA'] - sample_norm['CA/FA'])**2 +
                (row['W/B'] - sample_norm['W/B'])**2 +
                (row['SP'] - sample_norm['SP'])**2 +
                (row['Age'] - sample_norm['Age'])**2
            )
            distances.append((i, distance))
        
        # Sort by distance and get top N
        distances.sort(key=lambda x: x[1])
        similar_indices = [idx for idx, _ in distances[:top_n]]
        
        # Get the similar samples
        similar_samples = df.iloc[similar_indices]
        
        # Print results
        print("\n----- Most Similar Concrete Samples in Dataset -----")
        print("Your Sample:")
        print(f"Cement: {sample_params.get('cement', 0)} kg/m³")
        print(f"POFA: {sample_params.get('pofa', 0)} kg/m³")
        print(f"CA/FA: {sample_params.get('ca_fa', 0)}")
        print(f"W/B: {sample_params.get('w_b', 0)}")
        print(f"SP: {sample_params.get('sp', 0)} kg/m³")
        print(f"Age: {sample_params.get('age', 0)} days")
        
        print("\nSimilar Samples:")
        print(similar_samples[['C', 'POFA', 'CA/FA', 'W/B', 'SP', 'Age', 'CS']].to_string(index=False))
        
        # Calculate average CS of similar samples
        avg_cs = similar_samples['CS'].mean()
        print(f"\nAverage CS of similar samples: {avg_cs:.2f} MPa")
        
        return similar_samples
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    # Verify predictions on random samples
    results_df, model, scaler = verify_predictions(sample_count=20)
    
    # Example: Find similar samples for a specific concrete mix
    sample = {
        'cement': 400,
        'pofa': 100,
        'ca_fa': 1.30,
        'w_b': 0.35,
        'sp': 5.0,
        'age': 28
    }
    similar_samples = find_similar_samples(sample)
    
    # Predict for this sample
    if model is not None and scaler is not None:
        predicted_strength = predict_strength(
            model, scaler,
            sample['cement'],
            sample['pofa'],
            sample['ca_fa'],
            sample['w_b'],
            sample['sp'],
            sample['age']
        )
        print(f"\nPredicted CS for sample: {predicted_strength:.2f} MPa")