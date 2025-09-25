import pandas as pd
import os
from src.feature_engineering import FeatureCreation
from src.preprocessing import DataPreprocessing

df = pd.read_csv('data/preprocessed/preprocessed_airplane_price_dataset.csv')

# Feature Creation
feature_egnineering = FeatureCreation(df)
df = (
    feature_egnineering.create_Company()
      .create_HMC_per_person()
      .create_Cost_per_km()
      .change_Age()
      .getDataset()
)


preprocessing = DataPreprocessing(df)
df = (
    preprocessing.encoding()
    .scaling()
    .logTransformation()
    .getDataset()
)

# Save dataset
output_folder = 'data/engineered'
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'engineered_airplane_price_dataset.csv')
df.to_csv(output_path, index=False)

print(df.head(10))