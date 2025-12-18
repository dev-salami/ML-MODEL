"""
Main script for the enhanced POFA concrete strength prediction system.
Includes Bagging and GEP models with correlation analysis.
"""

import argparse
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import warnings

# Filter warnings
warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
np.random.seed(42)

# Import modules
from data_preprocessing import load_and_explore_data, preprocess_data
from utils import predict_strength

# Import individual model modules
import linear_regression
import random_forest
import gradient_boosting
import svr
import hybrid_model
import bagging_algorithm
import gep_model
import model_comparison
import correlation_analysis

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Enhanced POFA Concrete Strength Prediction')
    
    # Add model selection argument
    parser.add_argument('--model', type=str, 
                        choices=['linear', 'rf', 'gb', 'svr', 'hybrid', 'bagging', 'gep', 'all', 'compare', 'correlation'],
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
    
    elif model_type == 'bagging':
        model, _ = bagging_algorithm.train_bagging(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_model, _ = bagging_algorithm.tune_bagging(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_model, 'bagging_model_tuned.pkl')
        # Feature importance with error handling
        print(f"Number of features: {len(feature_names)}")
        print(f"Feature names: {feature_names}")
        try:
            bagging_algorithm.feature_importance_bagging(tuned_model, feature_names)
        except Exception as e:
            print(f"Warning: Could not generate feature importance plot: {e}")
        return tuned_model, scaler
    
    elif model_type == 'gep':
        best_ind, results, best_func = gep_model.train_gep(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        tuned_ind, tuned_results, tuned_func = gep_model.tune_gep(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names
        )
        # Save the GEP model (saving just the results and the picklable function)
        joblib.dump({
            'results': tuned_results,
            'model': tuned_func  # This is now a GEPModel instance that can be pickled
        }, 'gep_model.pkl')
        return tuned_func, scaler
        
    elif model_type == 'all':
        # Train all models
        
        # Original models
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
        
        # New models
        bagging_model, _ = bagging_algorithm.train_bagging(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        tuned_bagging_model, _ = bagging_algorithm.tune_bagging(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        joblib.dump(tuned_bagging_model, 'bagging_model_tuned.pkl')
        try:
            bagging_algorithm.feature_importance_bagging(tuned_bagging_model, feature_names)
        except Exception as e:
            print(f"Warning: Could not generate bagging feature importance plot: {e}")
        
        print("\nTraining GEP model (this may take a while)...")
        best_ind, results, best_func = gep_model.train_gep(
            X_train_scaled, y_train, X_test_scaled, y_test, feature_names, pop_size=200, n_gen=30
        )
        joblib.dump({
            'results': results,
            'model': best_func  # This is now a GEPModel instance
        }, 'gep_model.pkl')
        
        # Run correlation analysis
        correlation_analysis.generate_correlation_analysis(
            X_test_scaled, y_test, scaler, feature_names
        )
        
        # Return the tuned Bagging model as default
        return tuned_bagging_model, scaler
        
    elif model_type == 'compare':
        # Compare all available models
        model_comparison.compare_all_models()
        # Run correlation analysis
        correlation_analysis.generate_correlation_analysis(
            X_test_scaled, y_test, scaler, feature_names
        )
        
        # Return a default model
        if os.path.exists('bagging_model_tuned.pkl'):
            model = joblib.load('bagging_model_tuned.pkl')
            return model, scaler
        
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
    
    elif model_type == 'correlation':
        # Run correlation analysis only
        print("\nGenerating correlation analysis for all available models...")
        correlation_analysis.generate_correlation_analysis(
            X_test_scaled, y_test, scaler, feature_names
        )
        
        # Return Bagging or Random Forest model as default
        if os.path.exists('bagging_model_tuned.pkl'):
            model = joblib.load('bagging_model_tuned.pkl')
            return model, scaler
        
        if os.path.exists('random_forest_model_tuned.pkl'):
            model = joblib.load('random_forest_model_tuned.pkl')
            return model, scaler
        else:
            print("No saved model found for prediction. Training Random Forest model...")
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

def predict(model, scaler, args, model_type=None):
    """
    Make a prediction using the specified model and parameters.
    """
    # For GEP model
    if model_type == 'gep':
        # Make prediction using the GEP function
        try:
            # Scale the input data
            input_data = np.array([
                [args.cement, args.pofa, args.ca_fa, args.w_b, args.sp, args.age]
            ])
            input_data_scaled = scaler.transform(input_data)[0]
            
            # Apply the GEP function
            prediction = model(*input_data_scaled)
            
            # Handle invalid predictions
            if np.isnan(prediction) or np.isinf(prediction):
                print("Warning: GEP model produced an invalid prediction. Using fallback method.")
                prediction = predict_strength(
                    joblib.load('random_forest_model_tuned.pkl'), scaler,
                    args.cement, args.pofa, args.ca_fa, args.w_b, args.sp, args.age
                )
        except Exception as e:
            print(f"Error with GEP prediction: {e}. Using fallback method.")
            prediction = predict_strength(
                joblib.load('random_forest_model_tuned.pkl'), scaler,
                args.cement, args.pofa, args.ca_fa, args.w_b, args.sp, args.age
            )
    # For other scikit-learn models
    else:
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
        model_type = None
        
        if args.model == 'linear':
            model_file = 'linear_regression_model.pkl'
            model_type = 'sklearn'
        elif args.model == 'rf':
            model_file = 'random_forest_model_tuned.pkl'
            model_type = 'sklearn'
        elif args.model == 'gb':
            model_file = 'gradient_boosting_model_tuned.pkl'
            model_type = 'sklearn'
        elif args.model == 'svr':
            model_file = 'svr_model_tuned.pkl'
            model_type = 'sklearn'
        elif args.model == 'hybrid':
            model_file = 'hybrid_model.pkl'
            model_type = 'sklearn'
        elif args.model == 'bagging':
            model_file = 'bagging_model_tuned.pkl'
            model_type = 'sklearn'
        elif args.model == 'gep':
            model_file = 'gep_model.pkl'
            model_type = 'gep'
        else:
            # Default to Bagging or Random Forest model
            if os.path.exists('bagging_model_tuned.pkl'):
                model_file = 'bagging_model_tuned.pkl'
                model_type = 'sklearn'
            else:
                model_file = 'random_forest_model_tuned.pkl'
                model_type = 'sklearn'
        
        try:
            # Load model based on type
            if model_type == 'gep':
                gep_data = joblib.load(model_file)
                model = gep_data['model']  # This is now a GEPModel instance
            else:
                model = joblib.load(model_file)
            
            # Load scaler
            scaler = joblib.load('pofa_concrete_strength_scaler.pkl')
            
            # Make prediction
            predict(model, scaler, args, model_type)
            
        except FileNotFoundError:
            print(f"Model file {model_file} not found. Please train the model first.")
            model, scaler = train_model(args.model, args.data)
            predict(model, scaler, args, args.model)
    else:
        # Train the specified model
        model, scaler = train_model(args.model, args.data)
        
        # Make a prediction with the trained model
        predict(model, scaler, args, args.model)

if __name__ == "__main__":
    main()