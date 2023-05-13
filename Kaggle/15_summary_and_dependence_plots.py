# ==========================
# Summary & Dependence plots
# ==========================

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
import shap
import html

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

# Теория здесь https://www.kaggle.com/code/dansbecker/advanced-uses-of-shap-values/tutorial

data = pd.read_csv('datasets/fifa_2018.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# # # # #  Summary plot  # # # # #

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(val_X)

# When plotting, we call shap_values[1]. For classification problems, there is a separate array of
# SHAP values for each possible outcome. In this case, we index in to get the SHAP values for the prediction of "True".
shap.summary_plot(shap_values[1], val_X)

"""
This plot is made of many dots. Each dot has three characteristics:

Vertical location shows what feature it is depicting
Color shows whether that feature was high or low for that row of the dataset
Horizontal location shows whether the effect of that value caused a higher or lower prediction.
For example, the point in the upper left was for a team that scored few goals, reducing the prediction by 0.25.

Some things you should be able to easily pick out:

The model ignored the Red and Yellow & Red features.
Usually Yellow Card doesn't affect the prediction, but there is an extreme case where a high value caused a much lower prediction.
High values of Goal scored caused higher predictions, and low values caused low predictions

SHAP быстрее работает с XGB
"""

# # # # #  SHAP Dependence Contribution Plots  # # # # #

# calculate shap values. This is what we will plot.
shap_values = explainer.shap_values(X)

# make plot.
shap.dependence_plot('Ball Possession %', shap_values[1], X, interaction_index="Goal Scored")

"""
Start by focusing on the shape, and we'll come back to color in a minute. Each dot represents a row of the data.
The horizontal location is the actual value from the dataset, and the vertical location shows what having that value did
to the prediction. The fact this slopes upward says that the more you possess the ball, the higher the model's prediction
is for winning the Man of the Match award.
"""