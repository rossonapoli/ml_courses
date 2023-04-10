# =================
# DecisionTreeRegressor
# =================

import pandas as pd
from IPython.display import display
import sklearn
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

melbourne_file_path = '/Users/a.stepanenkov/PycharmProjects/Kaggle/melb_data.csv'

melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data.dropna(axis = 'index')

# Выбираем значение которое будем предсказывать
y = melbourne_data.Price

# Выбираем свойства объекта, на основе которых будет строиться модель
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
x = melbourne_data[melbourne_features]

# Одинаковый random state позволяет получить одинаковые результаты при каждом запуске
melbourne_model = DecisionTreeRegressor(random_state = 1)

# Строим модель
melbourne_model.fit(x, y)

print("Making predictions for the following 5 houses:")
print(x.head())
predictions = melbourne_model.predict(x.head())
print("The predictions are: ", predictions)
print("Actual prices are: ", y.head().to_list())

# Посчитаем MAE
predicted_home_prices = melbourne_model.predict(x)
print("MAE: ", mean_absolute_error(y, predicted_home_prices))

#Проблема: MAE занижена. См file_2