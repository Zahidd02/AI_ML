from main import test_data, test_set_label
import numpy as np
from sklearn.metrics import mean_squared_error
from joblib import load


# Here we will get the latest model which is RandomForestRegressor()
model = load('Real_Estate_Price_Predictor.joblib')
predicted_values = model.predict(test_data)
mse = mean_squared_error(test_set_label, predicted_values)
print(np.sqrt(mse))



