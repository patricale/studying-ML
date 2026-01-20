#Model-Based vs Instance-Based
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

#Download and Prepare the data
data_root = pd.read_csv("../data/system_metrics.csv")
df = pd.DataFrame(data_root)

X = df[['CPU_Usage', 'Fan_Speed']]
y = df[['Watt_Consumption']]

#Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Creation of the model
regr_model = LinearRegression()
knn_model = KNeighborsRegressor(n_neighbors=3)

#Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Fit the model
regr_model.fit(X_train, y_train)
knn_model.fit(X_train_scaled, y_train)

#Predict value
y_pred_regr = regr_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test_scaled)

#Evaluate models
MSE_regr = mean_squared_error(y_test, y_pred_regr)
MSE_knn = mean_squared_error(y_test, y_pred_knn)

r2_regr = r2_score(y_test, y_pred_regr)
r2_knn = r2_score(y_test, y_pred_knn)

print(f"MSE of the linear regression model: {MSE_regr:.3f}")
print(f"MSE of the KNN model: {MSE_knn:.3f}")

print(f"RMSE Linear Regression: {np.sqrt(MSE_regr):.3f}")
print(f"RMSE KNN: {np.sqrt(MSE_knn):.3f}")

print(f"R2 score for Linear Regression is: {r2_regr:.3f}")
print(f"R2 score for KNN is: {r2_knn:.3f}")

#Plot - Predictions vs Real values
plt.figure(figsize=(14, 5))

#1: Linear Regression
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_regr, alpha=0.6, color='blue', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title(f'Linear Regression\nR² = {r2_regr:.3f}, MSE = {MSE_regr:.3f}')
plt.grid(True, alpha=0.3)

# 2: KNN
plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_knn, alpha=0.6, color='green', edgecolors='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Real Values')
plt.ylabel('Predicted Values')
plt.title(f'K-Nearest Neighbors (k=3)\nR² = {r2_knn:.3f}, MSE = {MSE_knn:.3f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Comparison metrics
plt.figure(figsize=(10, 5))

# Comparison MSE
plt.subplot(1, 2, 1)
models = ['Linear Regression', 'KNN']
mse_values = [MSE_regr, MSE_knn]
colors = ['blue', 'green']
plt.bar(models, mse_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Mean Squared Error')
plt.title('Comparison MSE')
plt.grid(True, alpha=0.3, axis='y')

# Comparison R²
plt.subplot(1, 2, 2)
r2_values = [r2_regr, r2_knn]
plt.bar(models, r2_values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('R² Score')
plt.title('Comparison R² Score')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show() 