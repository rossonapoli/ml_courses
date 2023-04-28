# =======
# XGBoost
# =======

import matplotlib
import pandas as pd
from IPython.display import display
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

# Read the data
data = pd.read_csv('./melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

# Separate data into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

my_model = XGBRegressor(n_estimators = 1000, early_stopping_rounds = 5, learning_rate = 0.05, random_state = 0)

my_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)

'''
Важные параметры XGBRegressor
1. n_estimators
It specifies how many times to go through the modeling cycle described above.
It is equal to the number of models that we include in the ensemble.

- Too low a value causes underfitting, which leads to inaccurate predictions on both training data and test data.
- Too high a value causes overfitting, which causes accurate predictions on training data, but
inaccurate predictions on test data (which is what we care about).

2. early_stopping_rounds 
Останавливает модель когда после Х итераций МАЕ перестал уменьшаться. Х ставим на 5

3. learning_rate
Что-то типо веса получаемых в модели значений

4. n_jobs
Делает параллельные вычисления (ставим по количеству ядер). Имеет смысл только для больших датасетов
'''

predictions = my_model.predict(X_valid)
print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))