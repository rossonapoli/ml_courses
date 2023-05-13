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

data = pd.read_csv('./datasets/diabetes.csv')

y = data['Outcome']

# Удаляем столбец с survived
diabetes_predictors = data.drop(['Outcome'], axis=1)

# select_dtypes() method returns a new DataFrame that includes/excludes columns of the specified dtype
# В данном случае мы оставили только столбцы числовых типов
x = diabetes_predictors.select_dtypes(exclude=['object'])

# Divide data into training and validation subsets
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

# Поля с NaN значениям
cols_with_missing = [col for col in x_train.columns if x_train[col].isnull().any()]

# # # # # # # # 1. Пилим модель # # # # # # # #

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(x_train, y_train)
preds = model.predict(x_test)
print("Random tree MAE: " + str(mean_absolute_error(y_test, preds)))

# # # # # # # # 2. Проверяем Permutation Importance # # # # # # # #

perm = PermutationImportance(model, random_state=1).fit(x_test, y_test)
# Результат возвращается в виде html.
# The first number in each row shows how much model performance decreased with a random shuffling
# We measure the amount of randomness in our permutation importance calculation by repeating the
# process with multiple shuffles. The number after the ± measures how performance varied from one-reshuffling to the next.

w = eli5.show_weights(perm, feature_names = x_test.columns.tolist())
result = pd.read_html(w.data)[0]
print(result)

# # # # # # # # 3. Делаем PDP для атрибута который оказался важным # # # # # # # #

disp1 = PartialDependenceDisplay.from_estimator(model, x_test, ['Glucose'])
plt.show()

# # # # # # # # 4. Смотрим SHAP values # # # # # # # #

# Выбираем строку для анализа
row_to_show = 10
data_for_prediction = x_test.iloc[row_to_show]
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

# По идее data_for_prediction можно заменить на x_test, тогда SHAP values посчитаются для всех строк.
# Но shap.force_plot( matplotlib = True) не работает с несколькими строками, а без него вообще не показывается ничего
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_for_prediction)

# shap_values это список двух массивов. Первый из них для это SHAP values для позитивного исхода (нет диабета), второй -
# для негативного. В force_plot мы отправляем второй.
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib = True)

# # # # # # # # 5. Делаем Summary plot # # # # # # # #
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values[1], x_test)

# # # # # # # # 6. Делаем SHAP Dependence Contribution Plots # # # # # # # #
# Не самые логичные атрибуты для этого, но лучше не нашел
shap_values = explainer.shap_values(x)
shap.dependence_plot('BloodPressure', shap_values[1], x, interaction_index="Age")