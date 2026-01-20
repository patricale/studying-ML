#Model-Based 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
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
knn_model = KNeighborsRegressor(n_neighbors=1)

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

print(f"MSE of the linear regression model: {MSE_regr}")
print(f"MSE of the KNN model: {MSE_knn}")