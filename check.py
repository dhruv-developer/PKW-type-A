from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import pandas as pd


file_path = 'c1.xlsx'

# Read the Excel file
data = pd.read_excel(file_path)

# Display the first few rows to understand its structure
data.head()

# Remove unnecessary columns
cleaned_data = data[['ho', 'h1', 'do', 'ho/P', 'h1/P', 'Ho/P', 'Cdw']].dropna()

# Split the data into features and target
X = cleaned_data.drop('Cdw', axis=1)
y = cleaned_data['Cdw']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the SVM model
svm_model = SVR(kernel='rbf')
svm_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test)

# Calculate mean squared error
mse = mean_squared_error(y_test, y_pred)

mse, y_pred[:5], y_test[:5]  # Displaying MSE and first 5 predictions vs actual values
print(mse, y_pred[:5], y_test[:5])  # Displaying MSE and first 5 predictions vs actual values
