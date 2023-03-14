# (Code with Harry) https://www.youtube.com/watch?v=vKdgg8O2me8&list=PLu0W_9lII9ai6fAMHp-acBmJONT7Y4BSG&index=11
import numpy
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

# ['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'] : "Features"
diabetes = datasets.load_diabetes()
diabetes_x = diabetes.data  # Takes all the variables from the equation (x1, x2, x3,... from wo + w1x1 + w2x2 + ....)
diabetes_x_train = diabetes_x[:-30]  # Takes the training X-data except last 30 points.
diabetes_x_test = diabetes_x[-30:]  # Takes the remaining test X-data.

diabetes_y = diabetes.target  # Resultant value are the "labels" here
diabetes_y_train = diabetes_y[:-30]
diabetes_y_test = diabetes_y[-30:]

# Model creation and training
model = linear_model.LinearRegression()
model.fit(diabetes_x_train, diabetes_y_train)

# Testing the built model
diabetes_y_predicted = model.predict(diabetes_x_test)

print("Mean squared error is: ",
      mean_squared_error(diabetes_y_test, diabetes_y_predicted))  # Finding how much error is present from actual values
print("Weights: ", model.coef_)  # Resultant co-efficients from 'wo + w1x1 + w2x2 + ....', i.e. 'wo, w1, w2, ...'
print("Intercept: ", model.intercept_)  # 'y = mx + c', here the intercept is the 'c' part
