from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump

# Fetching the data
mnist = fetch_openml('mnist_784', parser='auto')
x, y = mnist.data, mnist.target  # data: features, target: labels

# Splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Training the classifier
model = RandomForestClassifier()  # Works best since it has a whopping 96% accuracy... :D
model.fit(x_train, y_train)
predicted_val = model.predict(x_test)
actual = [eval(i) for i in y_test]
prediction_list = [int(item) for item in predicted_val]
print(actual)
print(prediction_list)
print(accuracy_score(actual, prediction_list))

#dump(model, 'Handwritten-digit_Recognition.joblib')
