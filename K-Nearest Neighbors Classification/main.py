from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Loading the dataset
iris = datasets.load_iris()
features = iris.data  # Loading the features : ['sepal length', 'sepal width in cm', 'petal length', 'petal width']
labels = iris.target  # Loading the labels : ['Iris-Setosa', 'Iris-Versicolour', 'Iris-Virginica']

# Training the classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

predicted_value = clf.predict([[3.5, 8.2, 4.1, 3.3]])  # Predicting a random value
print(predicted_value)  # Answer : [1], i.e. Iris-Versicolour








