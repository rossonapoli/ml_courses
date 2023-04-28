# =======
# Leakage
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
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

# Read the data
data = pd.read_csv('./AER_credit_card_data.csv',
                   true_values = ['yes'], false_values = ['no'])

# Select target
y = data.card

# Select predictors
X = data.drop(['card'], axis=1)

# Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
cv_scores = cross_val_score(my_pipeline, X, y,
                            cv=5,
                            scoring='accuracy')

print("Cross-validation accuracy: %f" % cv_scores.mean())

# Изучив поля датасета можно задаться некоторыми вопросам.
# Например - expenditure mean expenditure on this card or on cards used before applying?
# At this point, basic data comparisons can be very helpful:

expenditures_cardholders = X.expenditure[y]
expenditures_noncardholders = X.expenditure[~y]

print('Fraction of those who received a card and had no expenditures: %.2f' \
    # Непонятно как работает expenditures_cardholders == 0
    %((expenditures_cardholders == 0).mean()))
print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
    %((expenditures_noncardholders == 0).mean()))

'''
As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received
a card had no expenditures. It's not surprising that our model appeared to have a high accuracy.
But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

Since share is partially determined by expenditure, it should be excluded too. The variables
active and majorcards are a little less clear, but from the description, they sound concerning.
In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.

We would run a model without target leakage as follows:
'''

# Drop leaky predictors from dataset
potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
X2 = X.drop(potential_leaks, axis=1)

# Evaluate the model with leaky predictors removed
cv_scores = cross_val_score(my_pipeline, X2, y,
                            cv=5,
                            scoring='accuracy')

print("Cross-val accuracy: %f" % cv_scores.mean())

# После убирания неподходящих полей точность понизилась до 83%, но для новых данных мы можем ожидать примерно такую же точность,
# тогда как предудыщий вариант имея точность 98% гораздо хуже работал бы с новыми данными
