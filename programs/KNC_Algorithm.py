# @title KNeighborsClassifier Algorithm
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Load data
iris = load_iris()
X, y = iris.data, iris.target

# 2. Split in train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create the model (K-Nearest Neighbors)
model = KNeighborsClassifier(n_neighbors=3)

# 4. Train the model
model.fit(X_train, y_train)

# 5. Evaluate the model
accuracy = model.score(X_test, y_test)
print(f"Accurancy: {accuracy:.2f}")