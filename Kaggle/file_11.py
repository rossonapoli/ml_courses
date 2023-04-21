# ======================
# Permutation Importance
# ======================

import matplotlib
import pandas as pd
from IPython.display import display, HTML
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
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import eli5
from eli5.sklearn import PermutationImportance
import lxml

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)


# Теория здесь https://www.kaggle.com/code/dansbecker/permutation-importance/tutorial

data = pd.read_csv('./fifa_2018.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Classifier а не Regressor потому что мы будем предсказывать не числовое значение
my_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0).fit(train_X, train_y)

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
# Результат возвращается в виде html.
# The first number in each row shows how much model performance decreased with a random shuffling
# We measure the amount of randomness in our permutation importance calculation by repeating the
# process with multiple shuffles. The number after the ± measures how performance varied from one-reshuffling to the next.

w = eli5.show_weights(perm, feature_names = val_X.columns.tolist())
result = pd.read_html(w.data)[0]
print(result)