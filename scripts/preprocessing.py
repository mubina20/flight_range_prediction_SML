import pandas as pd
import os
from src.preprocessing import DataPreprocessing

df = pd.read_csv('data/raw/airplane_price_dataset.csv')

preprocessing = DataPreprocessing(df)
df = preprocessing.fillMissingValues().getDataset()


# save
output_folder = 'data/preprocessed'
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'preprocessed_airplane_price_dataset.csv')
df.to_csv(output_path, index=False)

print(df.head(10))