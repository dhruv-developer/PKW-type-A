from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'c1.xlsx'
data = pd.read_excel(file_path)

# Display the structure and summary of the dataset
print("Dataset Overview:")
print(data.info())
print("\nDataset Description:")
print(data.describe())

# Visualize the data (optional)
print("\nGenerating pairplot for feature relationships...")
sns.pairplot(data[['ho', 'h1', 'do', 'ho/P', 'h1/P', 'Ho/P', 'Cdw']])
plt.savefig('pairplot.png')  # Save the plot if you're running it in a terminal without a GUI
print("Pairplot saved as 'pairplot.png'.\n")

# Clean and prepare the dataset
print("Cleaning data...")
cleaned_data = data[['ho', 'h1', 'do', 'ho/P', 'h1/P', 'Ho/P', 'Cdw']].dropna()
print(f"Cleaned data has {cleaned_data.shape[0]} rows and {cleaned_data.shape[1]} columns.\n")

# Define features and target
X = cleaned_data.drop('Cdw', axis=1)
y = cleaned_data['Cdw']

print("Features and Target defined:")
print("Features (X):")
print(X.head())
print("\nTarget (y):")
print(y.head())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nData split into training (80%) and testing (20%) sets.")
print(f"Training set: {X_train.shape[0]} rows, Testing set: {X_test.shape[0]} rows.\n")

# Scale the features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Feature scaling complete.\n")

# Use grid search to tune hyperparameters
print("Starting GridSearchCV for hyperparameter tuning...")
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

# Train and predict using the best model
print("\nTraining the best model...")
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)
print("Model training complete.\n")

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}\n")

# Compare predictions and actual values
comparison_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred})
print("\nSample Predictions (first 5 rows):")
print(comparison_df.head())

# Visualize predictions
print("\nGenerating scatter plot for Actual vs Predicted values...")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Identity line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted")
plt.savefig('actual_vs_predicted.png')  # Save the plot
print("Scatter plot saved as 'actual_vs_predicted.png'.\n")

print("Script execution completed.")
