# ========================================================================
# Тестирование и сравнение разных моделей (верификация на отдельном файле)
# ========================================================================

import pandas as pd
from IPython.display import display
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 2000)

# Сделаем то же что и в file_3 но для двух датасетов (один тренировочный, второй тестовый), а также потестим несколько моделей

# Read the data
X_full = pd.read_csv('/Users/a.stepanenkov/PycharmProjects/Kaggle/train.csv', index_col = 'Id')
X_test_full = pd.read_csv('/Users/a.stepanenkov/PycharmProjects/Kaggle/test.csv', index_col = 'Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Сплитуем только тренировочные данные. С тестовыми потом будем просто сверять
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size = 0.8, test_size = 0.2, random_state = 0)

# Делаем 5 разных моделей чтобы посмотреть какая лучше
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='absolute_error', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
# n_estimators - количество деревьев
# min_samples_split - The minimum number of samples required to split an internal node
# max_depth - максимальная глубина дерева

models = [model_1, model_2, model_3, model_4, model_5]

# Считаем МАЕ для каждой модели
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    # В модель отправляем тренировочные данные
    model.fit(X_t, y_t)
    # Предсказываем цену для тестовых (валидационных данных)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

for i in range(0, len(models)):
    mae = score_model(models[i])
    print("Model %d MAE: %d" % (i+1, mae))

# Теперь мы знаем что наименьшую МАЕ дает 3 модель
model_3.fit(X, y)

# Generate test predictions
preds_test = model_3.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': preds_test})

output.to_csv('submission.csv', index=False)