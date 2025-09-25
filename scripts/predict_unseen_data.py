import pandas as pd
from joblib import load
from sklearn.metrics import r2_score, mean_absolute_error
import os 

from src.preprocessing import DataPreprocessing
from src.feature_engineering import FeatureCreation

df = pd.read_csv('test/test_data.csv')

preprocessing = DataPreprocessing(df)
df = preprocessing.fillMissingValues().getDataset()

fc = FeatureCreation(df)
df = fc.change_Age().create_Company().create_Cost_per_km().create_HMC_per_person().getDataset()

df = preprocessing.encode().scale().logTransformation().getDataset()

x_test = df.drop(columns=["Range_(km)"], errors='ignore')
y_test = df["Range_(km)"] if "Range_(km)" in df.columns else None

model = load("models/LinearRegression.joblib")

y_pred = model.predict(x_test)

if y_test is not None:
    print("R2:", r2_score(y_test, y_pred))
    print("MAE:", mean_absolute_error(y_test, y_pred))

# os.makedirs("results", exist_ok=True)
# df["Prediction"] = y_pred
# df.to_csv("results/test_predictions.csv", index=False)