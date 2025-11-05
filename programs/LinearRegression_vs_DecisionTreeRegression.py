from sklearn import tree
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the data
data = fetch_california_housing()
x, y = data.data, data.target

# Split training and test data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Compute best depth
lr_model = LinearRegression()
dt_model = DecisionTreeRegressor(random_state=42)

lr_model.fit(x_train, y_train)
dt_model.fit(x_train, y_train)

lr_y_pred = lr_model.predict(x_test)
dt_y_pred = dt_model.predict(x_test)

lr_mse = mean_squared_error(y_test, lr_y_pred)
lr_r2 = r2_score(y_test, lr_y_pred)
dt_mse = mean_squared_error(y_test, dt_y_pred)
dt_r2 = r2_score(y_test, dt_y_pred)

print(f"Linear Regression - MSE: {lr_mse:.3f}, R²: {lr_r2:.3f}")
print(f"Decision Tree Regression - MSE: {dt_mse:.3f}, R²: {dt_r2:.3f}")

dt_r2_train = r2_score(y_train, dt_model.predict(x_train))
print(f"R² (coefficient of determination) on training set: {dt_r2_train:.3f}")

if(dt_r2_train > dt_r2):
  print(f"Possible overfitting!")
elif(dt_r2_train < dt_r2):
  print(f"Possible underfitting!")
else:
  print(f"Tree looks fine!")

# Plot the tree
plt.figure(figsize=(20,10))
tree.plot_tree(dt_model,
               feature_names=data.feature_names,
               filled=True,
               rounded=True,
               fontsize=8,
               max_depth=3)
plt.show()