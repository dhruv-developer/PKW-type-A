from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
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

# Convert datasets into CatBoost Pool (optional but recommended for CatBoost)
train_pool = Pool(X_train, y_train)
test_pool = Pool(X_test, y_test)

# Use grid search to tune hyperparameters for CatBoost
param_grid = {
    'iterations': [100, 500, 1000],
    'depth': [4, 6, 8],
    'learning_rate': [0.01, 0.1, 0.2],
    'l2_leaf_reg': [1, 3, 5, 7]
}
grid_search = GridSearchCV(
    CatBoostRegressor(random_state=42, silent=True),
    param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_
print("\nBest Parameters from Grid Search:")
print(grid_search.best_params_)

# Train and predict using the best model
best_model.fit(train_pool)
y_pred = best_model.predict(X_test)

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
plt.savefig('actual_vs_predicted_catboost.png')  # Save the plot
print("Scatter plot saved as 'actual_vs_predicted_catboost.png'.\n")

# Feature importance
feature_importances = best_model.feature_importances_
features = X.columns

# Visualize feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=features)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.savefig('feature_importance_catboost.png')  # Save the plot
print("Feature importance plot saved as 'feature_importance_catboost.png'.\n")

print("Script execution completed.")
