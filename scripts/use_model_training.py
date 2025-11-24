import pandas as pd

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost
from sklearn.model_selection import KFold, cross_val_score

from tabulate import tabulate
from src.model_training import Trainer
from src.feature_engineering import FeatureSelection
import os

df = pd.read_csv("data/preprocessed/preprocessed_dataset.csv")

fs = FeatureSelection(df, target="Range_(km)")
fs.filter_by_correlation()
selected_features = fs.get_selected_features()

x = df.drop('Range_(km)', axis=1)
y = df["Range_(km)"]

models = [
    LinearRegression(),
    Lasso(alpha=0.1),
    Ridge(alpha=0.0000001),
    ElasticNet(alpha=0.0001),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    ExtraTreesRegressor(random_state=42),
    HistGradientBoostingRegressor(random_state=42),
    SVR(kernel='linear', C=20.0),
    KNeighborsRegressor(n_neighbors=1),
    xgboost.XGBRegressor(),
    AdaBoostRegressor(n_estimators=200)
]

results = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model in models:
    trainer = Trainer(model, x, y)
    trainer.train().evaluate()
    trainer.save_model()

    cv_scores = cross_val_score(model, x, y, cv=kf, scoring='r2')

    results.append([model.__class__.__name__, trainer.r2, trainer.mae, cv_scores.mean(), cv_scores.std()])

headers = ["Algorithm", "r2_score", "mean_absolute_error", "K-Fold Mean", "K-Fold Std"]
best_model = max(results, key=lambda x: x[1])
worst_model = min(results, key=lambda x: x[1])

# ANSI codes
green = "\033[92m"
red = "\033[91m"
reset = "\033[0m"

for row in results:
    if row == best_model:
        row[:] = [green + str(i) + reset for i in row]
    elif row == worst_model:
        row[:] = [red + str(i) + reset for i in row]

table = tabulate(results, headers=headers, tablefmt="grid", floatfmt=".6f")

print(table)

def SaveComparison(table):
    with open('results/all_model_compare.txt', 'w') as f:
        f.write(table)

SaveComparison(table) 