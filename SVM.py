from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Load the dataset
file_path = 'c1.xlsx'
data = pd.read_excel(file_path)

# Display the structure and summary of the dataset
print("Dataset Overview:")
print(data.info())
print("\nDataset Description:")
print(data.describe())

# Visualize the data (optional)
sns.pairplot(data[['ho', 'h1', 'do', 'ho/P', 'h1/P', 'Ho/P', 'Cdw']])
plt.savefig('pairplot.png')  # Save the plot if needed
print("Pairplot saved as 'pairplot.png'.\n")

# Clean and prepare the dataset
cleaned_data = data[['ho', 'h1', 'do', 'ho/P', 'h1/P', 'Ho/P', 'Cdw']].dropna()

# Define features and target
X = cleaned_data.drop('Cdw', axis=1)
y = cleaned_data['Cdw']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Use grid search to tune hyperparameters
param_grid = {
    'kernel': ['rbf', 'linear', 'poly'],
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1]
}
grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_scaled, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# If the best kernel is linear, extract coefficients and intercept
if grid_search.best_params_['kernel'] == 'linear':
    print("\nExtracting equation for the linear kernel...")
    linear_model = SVR(kernel='linear', C=grid_search.best_params_['C'])
    linear_model.fit(X_train_scaled, y_train)
    
    # Extract coefficients and intercept
    coefficients = linear_model.coef_[0]
    intercept = linear_model.intercept_[0]
    feature_names = X.columns
    
    # Build the equation
    equation = f"Cdw = {intercept:.4f} "
    for i, coef in enumerate(coefficients):
        equation += f"+ ({coef:.4f} * {feature_names[i]}) "
    print("\nDerived Equation for Cdw:")
    print(equation)
    
    # Save the equation to a file
    with open("equation.txt", "w") as f:
        f.write(equation)
    print("\nEquation saved as 'equation.txt'.\n")
else:
    print("\nThe best model uses a non-linear kernel (e.g., RBF or poly), and an explicit equation cannot be extracted.")
    
    # Use SHAP for interpretability
    print("\nUsing SHAP to interpret feature importance for the non-linear kernel...")
    explainer = shap.KernelExplainer(best_model.predict, X_train_scaled)
    shap_values = explainer.shap_values(X_test_scaled, nsamples=100)
    
    # Visualize SHAP summary
    shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    plt.savefig("shap_summary.png")
    print("SHAP summary plot saved as 'shap_summary.png'.\n")

# Train and predict using the best model
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}\n")

# Compare predictions and actual values
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nSample Predictions (first 5 rows):")
print(comparison_df.head())

# Visualize predictions
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Identity line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.savefig('actual_vs_predicted.png')  # Save the plot
print("Scatter plot saved as 'actual_vs_predicted.png'.\n")

print("Script execution completed.")
