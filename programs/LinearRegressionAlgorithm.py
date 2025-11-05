from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the dataset
data = fetch_california_housing()
x, y = data.data[:, [0]], data.target

# 2. Split in train and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Create and train the model
model = LinearRegression()
model.fit(x_train, y_train)

# 4. Doing prevision
y_pred = model.predict(x_test)

# 5. Mean of real values
y_mean = np.mean(y_test)
print(f"Mean of real values (y_test): {y_mean:.3f}")

# 6. Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean squared error (MSE): {mse:.3f}")
print(f"R² (coefficient of determination): {r2:.3f}")

# Order data for the print
sort_idx = np.argsort(x_test[:, 0])
x_sorted = x_test[sort_idx]
y_pred_sorted = y_pred[sort_idx]

# 7. Graphic
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, alpha=0.5, label="Real Data", color="blue")
plt.plot(x_sorted, y_pred_sorted, color="red", linewidth=2, label="Regression Line")
plt.axhline(y_mean, color="green", linestyle="--", label="Data average (ȳ)")

plt.title("Relationship between average income and average house price")
plt.xlabel("MedInc")
plt.ylabel("MedHouseVal")
plt.legend()
plt.grid(True)
plt.show()