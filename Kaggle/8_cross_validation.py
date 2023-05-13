# ================
# Cross-validation
# ================
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

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)


# Read the data
data = pd.read_csv('datasets/melb_data.csv')

# Select subset of predictors
cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
X = data[cols_to_use]

# Select target
y = data.Price

my_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer(strategy = 'median')),
                              ('model', RandomForestRegressor(n_estimators=50, random_state=0))])

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores)

# Нарисовать график зависимости n_estimators и МАЕ

# Строка ниже делает так что график появляется прямо в окне Юпитера
# %matplotlib inline

# result - это словарь (n_estimators : МАЕ)
# plt.plot(list(results.keys()), list(results.values()))
# plt.show()