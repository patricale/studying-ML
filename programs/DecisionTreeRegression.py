from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load data
data = fetch_california_housing()
x, y = data.data, data.target

# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Creation and Train model
model = DecisionTreeRegressor(random_state=42)
model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Evaluation with MSE and R^2
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.3f}")
print(f"R²: {r2:.3f}")