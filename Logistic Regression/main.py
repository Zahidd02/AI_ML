# Training a logistic regression classifier to predict whether a flower is 'Iris-Virginica' or not
from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris['data'][:, 3:]  # Considering only the 3rd column of the dataset
y = (iris.target == 2).astype(int)

# Training a logistic regression classifier
clf = LogisticRegression()
clf.fit(x, y)
ans = clf.predict([[1.65911]])  # Predicts whether the value is 'Iris-Virginica' or not, i.e, 1 or 0
print(ans)

# Plotting the graph for the ML model
x_graph = np.linspace(0, 3, 1000).reshape(-1, 1)
y_prob = clf.predict_proba(x_graph)
plt.plot(x_graph, y_prob[:, 1], "b-")
plt.show()
