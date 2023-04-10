# =================
# Сплитуем датасет
# =================

import pandas as pd
from IPython.display import display
from sklearn.tree import DecisionTreeRegressor
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

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(train_X, train_y)

# Get predicted prices on validation data. МАЕ слишком большая
val_predictions = melbourne_model.predict(val_X)
print("Actual MAE based on comparison of training and validation data: ", mean_absolute_error(val_y, val_predictions))

# Влиять на МАЕ можно, в числе прочего, глубиной дерева (количеством объектом в листьях дерева).
# За это отвечает параметр DecisionTreeRegressor(max_leaf_nodes)

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Смотрим как отличается МАЕ в зависимости от max_leaf_nodes

maes = []

# Циклом смотрим минимальное МАЕ и лучшее max_leaf_nodes
for max_leaf_nodes in range(10, 5000, 100):
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    # print("Max leaf nodes: %d  \t\t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
    maes.append(my_mae)
    if int(my_mae) == 231576:
        print("Лучшее значение для max_leaf_nodes: ", max_leaf_nodes)
        print("Минимальная МАЕ: ", min(maes))
        break


