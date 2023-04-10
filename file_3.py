# =====================
# RandomForestRegressor
# =====================

import pandas as pd
from IPython.display import display
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

melbourne_file_path = '/Users/a.stepanenkov/PycharmProjects/Kaggle/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.dropna(axis = 'index')

y = melbourne_data.Price

melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(x, y, random_state = 0)

# Random forest - модель которая строит несколько деревьев и считает среднее между ними. МАЕ ниже.
forest_model = RandomForestRegressor(random_state = 1)
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))

