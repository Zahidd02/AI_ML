import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump

# Reading Excel file & setting print configurations; 'MEDV' is the label, rest of the columns are features.
orig_data = pd.read_excel('Housing data.xlsx')
housing_data = orig_data.copy()  # creating a copy of the original data to keep the original data safe.
pd.set_option('display.max_columns', 25)
pd.set_option('display.width', 250)

# In case of missing values in actual data do... (Not necessary in our data as there are no missing values)
# imputer = SimpleImputer(strategy="median")
# imputer.fit(housing_data)
# print(imputer.statistics_)  # prints the calculated median values
# imputer.transform(housing_data)

# Splitting data into test and train sets. "42" however, is the answer to everything in the universe :)
spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in spliter.split(housing_data, housing_data['CHAS']):
    test_set = housing_data.loc[test_index]
    train_set = housing_data.loc[train_index]

# Further splitting label column and features
train_set_label = train_set['MEDV']
train_set_features = train_set.drop(['MEDV'], axis=1)
test_set_label = test_set['MEDV']
test_set_features = test_set.drop(['MEDV'], axis=1)

# Trying out attributes combination
# housing_data["TAX/RM"] = housing_data["TAX"] / housing_data["RM"]  # Highly -ve co-relation value (obtained after checking the data and using "trail & error" method)


# Finding the co-relations between columns
# corr_matrix = housing_data.corr()
# print(corr_matrix['MEDV'].sort_values(ascending=False), end="\n\n\n")  # Here we find there is a strong correlation between ['MEDV', 'RM', 'ZN', 'LSTAT', 'TAX/RM'] for both positive and negative values
# attr = ['MEDV', 'RM', 'ZN', 'LSTAT', 'TAX/RM']
# scatter_matrix(housing_data[attr], alpha=0.8)
# housing_data.plot(kind="scatter", x="TAX/RM", y="MEDV", alpha=0.8)
# plt.show()
#plt.close()

# Creating a Pipeline to perform the operations one by one just like ".then()" in TS.
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),  # Not necessary (Used for missing data)
    ('std_scaler', StandardScaler())
])
train_data = my_pipeline.fit_transform(train_set_features)  # final numpy array for training set
test_data = my_pipeline.transform(test_set_features)  # final numpy array for test set (used in Tester.py)
# print(train_data_pipe)


# Selecting a desired model for ML
# 1. Simple Regression Model
model = LinearRegression()
model.fit(train_data, train_set_label)
predicted_values = model.predict(train_data[:5])
mse = mean_squared_error(train_set_label[:5], predicted_values)
rmse = np.sqrt(mse)

# Making cross validation for Simple Regression Model
scores = cross_val_score(model, train_data, train_set_label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# Noting the results in "Model Outputs.txt"
with open('Model Outputs.txt', 'a') as file:
    file.truncate(0)  # Truncates any previous record
    file.write("Model: Simple Regression\n" + "Scores: " + str(rmse_scores) + "\nMean: " + str(np.mean(rmse_scores)) + "\nStd. Deviation: " + str(np.std(rmse_scores)) + "\n\n")

# 2. Decision Tree Regression Model
model = DecisionTreeRegressor()
model.fit(train_data, train_set_label)
predicted_values = model.predict(train_data[:5])
mse = mean_squared_error(train_set_label[:5], predicted_values)
rmse = np.sqrt(mse)
# print(rmse)  # Getting "0.0" which is bad :(, result of "Overfitting"

# Making cross validation for Decision Tree Regression Model
scores = cross_val_score(model, train_data, train_set_label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# Noting the results in "Model Outputs.txt"
with open('Model Outputs.txt', 'a') as file:
    file.write("Model: Decision Tree Regression\n" + "Scores: " + str(rmse_scores) + "\nMean: " + str(np.mean(rmse_scores)) + "\nStd. Deviation: " + str(np.std(rmse_scores)) + "\n\n")


# 3. Random Forest Regression Model
model = RandomForestRegressor()
model.fit(train_data, train_set_label)
predicted_values = model.predict(train_data[:5])
mse = mean_squared_error(train_set_label[:5], predicted_values)

# Making cross validation for Random Forest Regression Model
scores = cross_val_score(model, train_data, train_set_label, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

# Noting the results in "Model Outputs.txt"
with open('Model Outputs.txt', 'a') as file:
    file.write("Model: Random Forest Regression\n" + "Scores: " + str(rmse_scores) + "\nMean: " + str(np.mean(rmse_scores)) + "\nStd. Deviation: " + str(np.std(rmse_scores)) + "\n\n")


# Saving the model by creating the joblib file
dump(model, 'Real_Estate_Price_Predictor.joblib')
