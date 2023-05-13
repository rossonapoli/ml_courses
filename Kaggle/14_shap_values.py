# ============
# SHAP values
# ============

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

# Теория здесь https://www.kaggle.com/code/dansbecker/shap-values/tutorial,
# https://towardsdatascience.com/one-feature-attribution-method-to-supposedly-rule-them-all-shapley-values-f3e04534983d

data = pd.read_csv('datasets/fifa_2018.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

prediction  = my_model.predict_proba(data_for_prediction_array)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

# The shap_values object above is a list with two arrays. The first array is the SHAP values for a negative
# outcome (don't win the award), and the second array is the list of SHAP values for the positive outcome (wins the award).
# We typically think about predictions in terms of the prediction of a positive outcome, so we'll pull out SHAP
# values for positive outcomes (pulling out shap_values[1]).
# It's cumbersome to review raw arrays, but the shap package has a nice way to visualize the results.

shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib = True)
# plt.savefig('summary_plot_result.jpg')

# If you look carefully at the code where we created the SHAP values, you'll notice we reference
# Trees in shap.TreeExplainer(my_model). But the SHAP package has explainers for every type of model.

# shap.DeepExplainer works with Deep Learning models.
# shap.KernelExplainer works with all models, though it is slower than other Explainers and it offers an
# approximation rather than exact Shap values.
# Here is an example using KernelExplainer to get similar results. The results
# aren't identical because KernelExplainer gives an approximate result. But the results tell the same story.

k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
# plt.savefig('summary_plot_result.jpg')

