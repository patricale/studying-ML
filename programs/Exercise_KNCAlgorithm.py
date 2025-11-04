from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# 1. Load data
wine = load_wine()
X, y = wine.data, wine.target

# 2. Split in train and test data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 3. Train the model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(x_train, y_train)

# 4. Give the data to classify at the model
y_pred = model.predict(x_test)

# 5. Calculate accurancy
accurancy = model.score(x_test, y_test)
print(f"Accurancy: {accurancy:.2f}")