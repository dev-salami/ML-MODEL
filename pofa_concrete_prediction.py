import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
# import xgboost as xgb
from sklearn.svm import SVR
import shap
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Read the data
# For now let's manually create the dataframe from the provided table
data = {
    'C': [], 'POFA': [], 'CA/FA': [], 'W/B': [], 'SP': [], 'Age': [], 'CS': []
}

# Adding data points from the document (rows 1-407)
# This would be your actual data from a CSV file
# For demonstration, I'll add a few sample rows

# Example of adding first 20 rows (you would add all 407 rows in practice)
data['C'].extend([1128, 1128, 1128, 1128, 936, 936, 936, 936, 789.6, 789.6, 
                  789.6, 789.6, 677, 677, 677, 677, 550, 550, 550, 550])
data['POFA'].extend([0, 0, 0, 0, 191.76, 191.76, 191.76, 191.76, 338, 338, 
                     338, 338, 451.2, 451.2, 451.2, 451.2, 0, 0, 0, 0])
data['CA/FA'].extend([2.99, 2.99, 2.99, 2.99, 2.99, 2.99, 2.99, 2.99, 2.99, 2.99,
                      2.99, 2.99, 2.99, 2.99, 2.99, 2.99, 1.39, 1.39, 1.39, 1.39])
data['W/B'].extend([0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.16,
                    0.16, 0.16, 0.16, 0.16, 0.16, 0.16, 0.27, 0.27, 0.27, 0.27])
data['SP'].extend([55, 55, 55, 55, 55, 55, 55, 55, 55, 55,
                   55, 55, 55, 55, 55, 55, 12.1, 12.1, 12.1, 12.1])
data['Age'].extend([3, 7, 28, 90, 3, 7, 28, 90, 3, 7, 
                    28, 90, 3, 7, 28, 90, 1, 3, 7, 28])
data['CS'].extend([131.4, 136, 138, 139, 144, 148, 154, 156, 136, 142,
                  146, 149, 131.8, 135, 143, 145, 63, 78.8, 86.9, 91.4])

# In real implementation, you would load from CSV
df = pd.read_csv('pofa_concrete_data.csv')

 
# Create DataFrame from the manual data
#df = pd.DataFrame(data)


# 1. Data Exploration
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
plt.savefig('feature_histograms.png')
plt.close()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
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
plt.savefig('feature_vs_cs.png')
plt.close()

# 2. Data Preprocessing
X = df.drop('CS', axis=1)
y = df['CS']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Building and Evaluation

# Function to evaluate model performance
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
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

# Create a dictionary to store model results
model_results = {}

# Linear Regression
lr = LinearRegression()
lr_model, lr_pred, lr_mse, lr_r2 = evaluate_model(lr, X_train_scaled, X_test_scaled, y_train, y_test, "Linear Regression")
model_results['Linear Regression'] = {'model': lr_model, 'mse': lr_mse, 'r2': lr_r2}

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf_model, rf_pred, rf_mse, rf_r2 = evaluate_model(rf, X_train_scaled, X_test_scaled, y_train, y_test, "Random Forest")
model_results['Random Forest'] = {'model': rf_model, 'mse': rf_mse, 'r2': rf_r2}

# Gradient Boosting
gb = GradientBoostingRegressor(random_state=42)
gb_model, gb_pred, gb_mse, gb_r2 = evaluate_model(gb, X_train_scaled, X_test_scaled, y_train, y_test, "Gradient Boosting")
model_results['Gradient Boosting'] = {'model': gb_model, 'mse': gb_mse, 'r2': gb_r2}

# XGBoost
# xgb_model = xgb.XGBRegressor(random_state=42)
# xgb_model, xgb_pred, xgb_mse, xgb_r2 = evaluate_model(xgb_model, X_train_scaled, X_test_scaled, y_train, y_test, "XGBoost")
# model_results['XGBoost'] = {'model': xgb_model, 'mse': xgb_mse, 'r2': xgb_r2}

# Support Vector Regressor
svr = SVR()
svr_model, svr_pred, svr_mse, svr_r2 = evaluate_model(svr, X_train_scaled, X_test_scaled, y_train, y_test, "SVR")
model_results['SVR'] = {'model': svr_model, 'mse': svr_mse, 'r2': svr_r2}

# Find the best model based on R²
best_model_name = max(model_results.items(), key=lambda x: x[1]['r2'])[0]
print(f"Best Model: {best_model_name}")

# 4. Hyperparameter Tuning for the best model
if best_model_name == 'Random Forest':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestRegressor(random_state=42)
elif best_model_name == 'Gradient Boosting':
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0]
    }
    best_model = GradientBoostingRegressor(random_state=42)
# elif best_model_name == 'XGBoost':
#     param_grid = {
#         'n_estimators': [50, 100, 200],
#         'learning_rate': [0.01, 0.1, 0.2],
#         'max_depth': [3, 5, 7],
#         'subsample': [0.8, 0.9, 1.0],
#         'colsample_bytree': [0.8, 0.9, 1.0]
#     }
#     best_model = xgb.XGBRegressor(random_state=42)
else:  # Default to RandomForest if other models are best
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    best_model = RandomForestRegressor(random_state=42)

# Grid search with cross-validation
grid_search = GridSearchCV(best_model, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train_scaled, y_train)

# Best hyperparameters
print("Best Hyperparameters:")
print(grid_search.best_params_)

# Evaluate the tuned model
tuned_model = grid_search.best_estimator_
tuned_model.fit(X_train_scaled, y_train)
y_pred_tuned = tuned_model.predict(X_test_scaled)

mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = np.sqrt(mse_tuned)
r2_tuned = r2_score(y_test, y_pred_tuned)
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)

print(f"Tuned {best_model_name} Performance:")
print(f"MSE: {mse_tuned:.4f}")
print(f"RMSE: {rmse_tuned:.4f}")
print(f"R²: {r2_tuned:.4f}")
print(f"MAE: {mae_tuned:.4f}")

# 5. SHAP Analysis for Feature Importance
# Create a SHAP explainer
# if 'Random Forest' in best_model_name or 'XGBoost' in best_model_name or 'Gradient Boosting' in best_model_name:
#     explainer = shap.TreeExplainer(tuned_model)
#     # Calculate SHAP values
#     shap_values = explainer.shap_values(X_test_scaled)
    
#     # Summary plot
#     plt.figure(figsize=(10, 8))
#     shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
#     plt.title('SHAP Summary Plot')
#     plt.tight_layout()
#     plt.savefig('shap_summary.png')
#     plt.close()
    
#     # Feature importance based on SHAP
#     plt.figure(figsize=(10, 8))
#     shap_importance = np.abs(shap_values).mean(0)
#     feature_importance = pd.DataFrame(list(zip(X.columns, shap_importance)), 
#                                       columns=['Feature', 'SHAP Importance'])
#     feature_importance.sort_values(by=['SHAP Importance'], ascending=False, inplace=True)
    
#     sns.barplot(x='SHAP Importance', y='Feature', data=feature_importance)
#     plt.title('Feature Importance based on SHAP Values')
#     plt.tight_layout()
#     plt.savefig('shap_feature_importance.png')
#     plt.close()
    
#     # Individual SHAP value plots for top 3 features
#     for feature in feature_importance['Feature'][:3]:
#         plt.figure(figsize=(10, 6))
#         feature_idx = list(X.columns).index(feature)
#         shap.dependence_plot(feature_idx, shap_values, X_test, feature_names=X.columns, show=False)
#         plt.title(f'SHAP Dependence Plot for {feature}')
#         plt.tight_layout()
#         plt.savefig(f'shap_dependence_{feature}.png')
#         plt.close()

# 6. Advanced Hybrid Model (Optional)
# Create a stacking ensemble model
from sklearn.ensemble import StackingRegressor

estimators = [
    ('rf', RandomForestRegressor(random_state=42)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    # ('xgb', xgb.XGBRegressor(random_state=42))
]

stacking_model = StackingRegressor(
    estimators=estimators,
    final_estimator=SVR()
)

# Cross-validation for the hybrid model
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(stacking_model, X_train_scaled, y_train, cv=kf, scoring='r2')

print("Hybrid Model (Stacking) Cross-Validation Results:")
print(f"Average R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# Final evaluation of the hybrid model
stacking_model.fit(X_train_scaled, y_train)
y_pred_hybrid = stacking_model.predict(X_test_scaled)

mse_hybrid = mean_squared_error(y_test, y_pred_hybrid)
rmse_hybrid = np.sqrt(mse_hybrid)
r2_hybrid = r2_score(y_test, y_pred_hybrid)
mae_hybrid = mean_absolute_error(y_test, y_pred_hybrid)

print(f"Hybrid Model Performance:")
print(f"MSE: {mse_hybrid:.4f}")
print(f"RMSE: {rmse_hybrid:.4f}")
print(f"R²: {r2_hybrid:.4f}")
print(f"MAE: {mae_hybrid:.4f}")

# 7. Visualization of Results
plt.figure(figsize=(10, 6))
models = ['Linear Regression', 'Random Forest', 'Gradient Boosting',  'SVR', 'Tuned Model', 'Hybrid Model']
r2_scores = [lr_r2, rf_r2, gb_r2, svr_r2, r2_tuned, r2_hybrid]
plt.bar(models, r2_scores)
plt.xlabel('Models')
plt.ylabel('R² Score')
plt.title('Model Comparison - R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.close()

# 8. Create a prediction function for new data
# def predict_strength(model, scaler, cement, pofa, ca_fa, w_b, sp, age):
#     """
#     Predict concrete compressive strength using the trained model.
    
#     Parameters:
#     -----------
#     model: The trained ML model
#     scaler: The feature scaler
#     cement: Cement content (kg/m³)
#     pofa: Palm Oil Fuel Ash content (kg/m³)
#     ca_fa: Coarse Aggregate to Fine Aggregate ratio
#     w_b: Water to Binder ratio
#     sp: Superplasticizer content (kg/m³)
#     age: Age of concrete (days)
    
#     Returns:
#     --------
#     float: Predicted compressive strength (MPa)
#     """
#     # Create input data
#     input_data = np.array([[cement, pofa, ca_fa, w_b, sp, age]])
    
#     # Scale the input data
#     scaled_input = scaler.transform(input_data)
    
#     # Make prediction
#     prediction = model.predict(scaled_input)
    
#     return prediction[0]

# # Example prediction
# new_concrete = {
#     'cement': 400,
#     'pofa': 100,
#     'ca_fa': 1.30,
#     'w_b': 0.35,
#     'sp': 5.0,
#     'age': 28
# }

# predicted_strength = predict_strength(
#     tuned_model, scaler,
#     new_concrete['cement'],
#     new_concrete['pofa'],
#     new_concrete['ca_fa'],
#     new_concrete['w_b'],
#     new_concrete['sp'],
#     new_concrete['age']
# )

# print(f"\nPredicted Compressive Strength for New Concrete: {predicted_strength:.2f} MPa")

def predict_strength(model, scaler, cement, pofa, ca_fa, w_b, sp, age, additional_feature=0):
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
    additional_feature: Value for the 7th feature in your dataset (modify name as needed)
    
    Returns:
    --------
    float: Predicted compressive strength (MPa)
    """
    # Create input data with all 7 features
    input_data = np.array([[cement, pofa, ca_fa, w_b, sp, age, additional_feature]])
    
    # Scale the input data
    scaled_input = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    return prediction[0]

# Fixed example usage:
new_concrete = {
    'cement': 400,
    'pofa': 100,
    'ca_fa': 1.30,
    'w_b': 0.35,
    'sp': 5.0,
    'age': 90,
    'additional_feature': 0  # Replace with appropriate value or parameter name
}

# Call the function with all parameters
predicted_strength = predict_strength(
    tuned_model, scaler,
    new_concrete['cement'],
    new_concrete['pofa'],
    new_concrete['ca_fa'],
    new_concrete['w_b'],
    new_concrete['sp'],
    new_concrete['age'],
    new_concrete['additional_feature']  # Include the 7th feature
)

print(f"\nPredicted Compressive Strength for New Concrete: {predicted_strength:.2f} MPa")

# Alternative approach: Inspect what features the scaler was trained on
def identify_missing_feature():
    # Load the saved scaler (if available)
    try:
        loaded_scaler = joblib.load('pofa_concrete_strength_scaler.pkl')
        # Print feature names if available (scikit-learn 1.0+)
        if hasattr(loaded_scaler, 'feature_names_in_'):
            print("Scaler was trained on these features:", loaded_scaler.feature_names_in_)
        # Print the expected shape
        print(f"Scaler expects input with shape: (n_samples, {loaded_scaler.n_features_in_})")
    except:
        print("Could not load scaler or scaler doesn't have feature information")
    
    # Load the original CSV to inspect columns
    try:
        df_original = pd.read_csv('pofa_concrete_data.csv')
        print("Original CSV columns:", df_original.columns.tolist())
        print(f"Original CSV has {len(df_original.columns) - 1} features (excluding target)")
    except:
        print("Could not load original CSV file")

# You can call this function to debug:
# identify_missing_feature()

# 9. Save the model for future use
import joblib

# Save the model
joblib.dump(tuned_model, 'pofa_concrete_strength_model.pkl')
# Save the scaler
joblib.dump(scaler, 'pofa_concrete_strength_scaler.pkl')

print("\nModel and scaler saved successfully.")