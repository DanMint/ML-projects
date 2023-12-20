import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

model_filename = 'linear_regression_model.joblib'
model = joblib.load(model_filename)

file_path = 'data\\all_data.csv'
train_data = pd.read_csv(file_path)

# Preprocessing: Converting categorical data to numerical
train_data_processed = pd.get_dummies(train_data, columns=['Neighborhood'])

# Separating the features and the target variable
X = train_data_processed.drop('Price', axis=1)
y = train_data_processed['Price']

# Use the model to make predictions
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
rmse = mean_squared_error(y, y_pred, squared=False)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared (R²): {r2}")
