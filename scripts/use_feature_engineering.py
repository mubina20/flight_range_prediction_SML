import pandas as pd
import os
from src.feature_engineering import FeatureCreation

df = pd.read_csv('data/raw/airplane_dataset.csv')

# Feature Creation
feature_egnineering = FeatureCreation(df)
df = feature_egnineering.create_features().getDataset()

# Save dataset
output_folder = 'data/engineered'
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'engineered_dataset.csv')
df.to_csv(output_path, index=False)

print(df.head(10))