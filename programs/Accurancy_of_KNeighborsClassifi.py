# @title Accurancy of KNeighborsClassifier Algorithm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# 1. Load and Split data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Calculated the accurancy for differents k values
accuracies = []
k_values = range(1, 21)  # from 1 to 20 

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

# 3. Plot the ghraphic
plt.figure(figsize=(8, 4))
plt.plot(k_values, accuracies, marker='o')
plt.title("Accurancy as k varies")
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accurancy")
plt.grid(True)
plt.show()

# 4. Print of accurancy value
for k, acc in zip(k_values, accuracies):
    print(f"k={k:2d} → accurancy = {acc:.2f}")
