# ==============================
# Partial dependence plots (PDP)
# ==============================

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
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz
from matplotlib import pyplot as plt
from sklearn.inspection import PartialDependenceDisplay

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

# Теория здесь https://www.kaggle.com/code/dansbecker/partial-plots

data = pd.read_csv('./fifa_2018.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

# Строка ниже создает граф где показаны исходы при различных значениях изменяемой переменной
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
a = graphviz.Source(tree_graph, format="png")
# a.view()

# # # # #  1-feature Partial Dependence Plots  # # # # #

# Create and plot the data
# The y axis is interpreted as change (!!!!!) in the prediction from what it would be predicted at the baseline or leftmost value.
# From this particular graph, we see that scoring a goal substantially increases your chances of winning "Man of The Match."
# But extra goals beyond that appear to have little impact on predictions.
disp1 = PartialDependenceDisplay.from_estimator(tree_model, val_X, ['Goal Scored'])
plt.show()

# Другой пример
feature_to_plot = 'Distance Covered (Kms)'
disp2 = PartialDependenceDisplay.from_estimator(tree_model, val_X, [feature_to_plot])
plt.show()

# This graph seems too simple to represent reality. But that's because the model is so simple.
# You should be able to see from the decision tree above that this is representing exactly the model's structure.
# You can easily compare the structure or implications of different models. Here is the same plot with a Random Forest model.

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# kind='individual' показывает все результаты а не только усредненное
disp3 = PartialDependenceDisplay.from_estimator(rf_model, val_X, [feature_to_plot], kind='individual')
plt.show()

# # # # #  2-features Partial Dependence Plots  # # # # #

# We will again use the Decision Tree model for this graph. It will create an extremely simple plot,
# but you should be able to match what you see in the plot to the tree itself.

fig, ax = plt.subplots(figsize=(8, 6))
f_names = [('Goal Scored', 'Distance Covered (Kms)')]
# Similar to previous PDP plot except we use tuple of features instead of single feature
disp4 = PartialDependenceDisplay.from_estimator(rf_model, val_X, f_names, ax=ax)
plt.show()

# # # # #  3-features Partial Dependence Plots  # # # # #