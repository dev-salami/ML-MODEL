import argparse
import joblib
import os
import numpy as np

from data_preprocessing import load_and_explore_data, preprocess_data
from utils import predict_strength

# Import individual model modules
import linear_regression
import random_forest
import gradient_boosting
import svr
import hybrid_model
import model_comparison

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='POFA Concrete Strength Prediction')
    
    # Add model selection argument
    parser.add_argument('--model', type=str, choices=['linear', 'rf', 'gb', 'svr', 'hybrid', 'all', 'compare'],
                        default='all', help='Model to train (default: all)')
    
    # Add data file argument
    parser.add_argument('--data', type=str, default='pofa_concrete_data.csv',
                        help='Path to the CSV data file (default: pofa_concrete_data.csv)')
    
    # Add prediction mode argument
    parser.add_argument('--predict', action='store_true',
                        help='Predict using saved model instead of training')
    
    # Add prediction parameters
    parser.add_argument('--cement', type=float, default=400,
                        help='Cement content (kg/m³) for prediction')
    parser.add_argument('--pofa', type=float, default=100,
                        help='Palm Oil Fuel Ash content (kg/m³) for prediction')
    parser.add_argument('--ca_fa', type=float, default=1.30,
                        help='Coarse Aggregate to Fine Aggregate ratio for prediction')
    parser.add_argument('--w_b', type=float, default=0.35,
                        help='Water to Binder ratio for prediction')
    parser.add_argument('--sp', type=float, default=5.0,
                        help='Superplasticizer content (kg/m³) for prediction')
    parser.add_argument('--age', type=int, default=28,
                        help='Age of concrete (days) for prediction')
    
    return parser.parse_args()

def train_model(model_type, data_file):
    """
    Train a specific model type.
    """
    # Load and preprocess data
    df = load_and_explore_data(data_file)
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'pofa_concrete_strength_scaler.pkl')
    
    if model_type == 'linear':
        model, _ = linear_regression.train_linear_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(model, 'linear_regression_model.pkl')
        return model, scaler
        
    elif model_type == 'rf':
        model, _ = random_forest.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_model, _ = random_forest.tune_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_model, 'random_forest_model_tuned.pkl')
        return tuned_model, scaler
        
    elif model_type == 'gb':
        model, _ = gradient_boosting.train_gradient_boosting(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_model, _ = gradient_boosting.tune_gradient_boosting(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_model, 'gradient_boosting_model_tuned.pkl')
        return tuned_model, scaler
        
    elif model_type == 'svr':
        model, _ = svr.train_svr(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_model, _ = svr.tune_svr(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_model, 'svr_model_tuned.pkl')
        return tuned_model, scaler
        
    elif model_type == 'hybrid':
        model, _ = hybrid_model.train_hybrid_model(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(model, 'hybrid_model.pkl')
        return model, scaler
        
    elif model_type == 'all':
        # Train all models
        linear_model, _ = linear_regression.train_linear_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(linear_model, 'linear_regression_model.pkl')
        
        rf_model, _ = random_forest.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_rf_model, _ = random_forest.tune_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_rf_model, 'random_forest_model_tuned.pkl')
        
        gb_model, _ = gradient_boosting.train_gradient_boosting(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_gb_model, _ = gradient_boosting.tune_gradient_boosting(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_gb_model, 'gradient_boosting_model_tuned.pkl')
        
        svr_model, _ = svr.train_svr(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_svr_model, _ = svr.tune_svr(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_svr_model, 'svr_model_tuned.pkl')
        
        hybrid_model_trained, _ = hybrid_model.train_hybrid_model(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(hybrid_model_trained, 'hybrid_model.pkl')
        
        # Return the tuned Random Forest model as default
        return tuned_rf_model, scaler
        
    elif model_type == 'compare':
        # Compare all models and return
        model_comparison.compare_all_models()
        # Load the best model (assuming it's the tuned Random Forest for simplicity)
        if os.path.exists('random_forest_model_tuned.pkl'):
            model = joblib.load('random_forest_model_tuned.pkl')
            return model, scaler
        else:
            print("No saved model found. Training Random Forest model...")
            rf_model, _ = random_forest.train_random_forest(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            tuned_model, _ = random_forest.tune_random_forest(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            joblib.dump(tuned_model, 'random_forest_model_tuned.pkl')
            return tuned_model, scaler
    
    # Default case
    print("Invalid model type. Training Random Forest model...")
    rf_model, _ = random_forest.train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    tuned_model, _ = random_forest.tune_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    joblib.dump(tuned_model, 'random_forest_model_tuned.pkl')
    return tuned_model, scaler

def predict(model, scaler, args):
    """
    Make a prediction using the specified model and parameters.
    """
    prediction = predict_strength(
        model, scaler,
        args.cement, args.pofa, args.ca_fa, args.w_b, args.sp, args.age
    )
    
    print("\nPrediction Parameters:")
    print(f"Cement: {args.cement} kg/m³")
    print(f"POFA: {args.pofa} kg/m³")
    print(f"CA/FA: {args.ca_fa}")
    print(f"W/B: {args.w_b}")
    print(f"SP: {args.sp} kg/m³")
    print(f"Age: {args.age} days")
    print(f"\nPredicted Compressive Strength: {prediction:.2f} MPa")

def main():
    """
    Main function to run the program.
    """
    # Parse command-line arguments
    args = parse_arguments()
    
    if args.predict:
        # Load the model and scaler for prediction
        model_file = None
        if args.model == 'linear':
            model_file = 'linear_regression_model.pkl'
        elif args.model == 'rf':
            model_file = 'random_forest_model_tuned.pkl'
        elif args.model == 'gb':
            model_file = 'gradient_boosting_model_tuned.pkl'
        elif args.model == 'svr':
            model_file = 'svr_model_tuned.pkl'
        elif args.model == 'hybrid':
            model_file = 'hybrid_model.pkl'
        else:
            # Default to Random Forest model
            model_file = 'random_forest_model_tuned.pkl'
        
        try:
            model = joblib.load(model_file)
            scaler = joblib.load('pofa_concrete_strength_scaler.pkl')
            predict(model, scaler, args)
        except FileNotFoundError:
            print(f"Model file {model_file} not found. Please train the model first.")
            model, scaler = train_model(args.model, args.data)
            predict(model, scaler, args)
    else:
        # Train the specified model
        model, scaler = train_model(args.model, args.data)
        
        # Make a prediction with the trained model
        predict(model, scaler, args)

if __name__ == "__main__":
    main()