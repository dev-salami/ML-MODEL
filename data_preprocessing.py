import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')
from utils import OUTPUT_DIRS

def load_and_explore_data(file_path='pofa_concrete_data.csv'):
    """
    Load, explore and visualize the concrete dataset.
    
    Parameters:
    -----------
    file_path: Path to the CSV file
    
    Returns:
    --------
    df: Pandas DataFrame containing the data
    """
    # Read the data
    df = pd.read_csv(file_path)
    
    # Data Summary
    print("Data Summary:")
    print(df.describe())
    
    # Check for missing values
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Data Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot histograms for each feature
    for i, column in enumerate(df.columns):
        plt.subplot(3, 3, i+1)
        plt.hist(df[column], bins=20)
        plt.title(column)
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS['misc'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['misc'], 'feature_histograms.png'))
    plt.close()
    
    # Correlation matrix
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    os.makedirs(OUTPUT_DIRS['correlation'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['correlation'], 'correlation_matrix.png'))
    plt.close()
    
    # Scatter plots for important relationships
    plt.figure(figsize=(20, 15))
    features = ['C', 'POFA', 'CA/FA', 'W/B', 'SP', 'Age']
    for i, feature in enumerate(features):
        plt.subplot(3, 2, i+1)
        plt.scatter(df[feature], df['CS'])
        plt.xlabel(feature)
        plt.ylabel('Compressive Strength (CS)')
        plt.title(f'{feature} vs CS')
    plt.tight_layout()
    os.makedirs(OUTPUT_DIRS['correlation'], exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIRS['correlation'], 'feature_vs_cs.png'))
    plt.close()
    
    return df

def preprocess_data(df):
    """
    Preprocess the data by splitting into train/test sets and scaling features.
    
    Parameters:
    -----------
    df: Pandas DataFrame with the data
    
    Returns:
    --------
    X_train_scaled: Scaled training features
    X_test_scaled: Scaled test features
    y_train: Training target values
    y_test: Test target values
    scaler: Fitted StandardScaler object
    X.columns: Feature names
    """
    # Separate features and target
    X = df.drop('CS', axis=1)  # Features (6 features, excluding CS)
    y = df['CS']               # Target variable
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Print information about the scaling for debugging
    n_features = scaler.n_features_in_ if hasattr(scaler, 'n_features_in_') else "Unknown"
    print(f"Scaler was trained with {n_features} features")
    if hasattr(scaler, 'feature_names_in_'):
        print(f"Feature names: {scaler.feature_names_in_}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

if __name__ == "__main__":
    # Load and explore data
    df = load_and_explore_data()
    
    # Preprocess data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # Save the scaler for future use
    os.makedirs(OUTPUT_DIRS['models'], exist_ok=True)
    joblib.dump(scaler, os.path.join(OUTPUT_DIRS['models'], 'pofa_concrete_strength_scaler.pkl'))
    
    print("Data preprocessing completed and scaler saved.")