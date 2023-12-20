import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

file_path = 'data\\all_data.csv'
train_data = pd.read_csv(file_path)

# Preprocessing: Converting categorical data to numerical
train_data_processed = pd.get_dummies(train_data, columns=['Neighborhood'])

# Separating the features and the target variable
X = train_data_processed.drop('Price', axis=1)
y = train_data_processed['Price']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating and training the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on the test set and calculating the mean squared error
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)

# Saving the model to a file
model_filename = 'linear_regression_model.joblib'
joblib.dump(model, model_filename)

print(f"Mean Squared Error: {mse}")
print(f"Model saved at: {model_filename}")
