import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree

np.random.seed(42)
X = np.sort(np.random.rand(100, 1) * 10, axis = 0)  
y = X.flatten() ** 2 + np.random.randn(100) * 5  

model = DecisionTreeRegressor(max_depth = 3)
model.fit(X, y)

X_test = np.linspace(0, 10, 500).reshape(-1, 1)
y_pred = model.predict(X_test)

plt.scatter(X, y, label = "Training Data", color = "blue")
plt.plot(X_test, y_pred, label = "Decision Tree Prediction", color = "red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Decision Tree Regression Ex")
plt.legend()
plt.show()

plt.figure(figsize = (12, 6))
plot_tree(model, filled = True, feature_names = ["X"], rounded = True)
plt.show()
