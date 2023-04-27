# =======
# 3D PDP
# =======

# ЧТО-ТО ИЗ ЭТОГО ЛОМАЕТ ИНТЕРПРЕТАТОР!!!!!!!!!!!

from pdpbox.pdp import pdp_interact, pdp_interact_plot, pdp_isolate, pdp_plot
import plotly.graph_objects as go
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np

data = pd.read_csv('./fifa_2018.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
tree_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

def pdp_3d(feature1, feature2, test_data, estimator, target, title):
    features_2 = [feature1, feature2]
    interaction = pdp_interact(model = estimator,
        dataset = test_data,
        model_features = test_data.columns,
        features = features_2)

    pdp = interaction.pdp.pivot_table(
        values = 'preds',
        columns = features_2[0],
        index = features_2[1])[::1]

    surface = go.Surface(
        x = pdp.columns,
        y = pdp.index,
        z = pdp.values)

    layout = go.Layout(
        # scene = dict(
        #     xaxis = dict(title = features_2[0]),
        #     yaxis = dict(title = features_2[1]),
        #     zaxis = dict(title = target), ),
            # title = 'gerg'
        )

    fig = go.Figure(surface, layout)
    fig.show()

pdp_3d('Goal Scored', 'Distance Covered (Kms)', val_X, tree_model, y, 'erget')