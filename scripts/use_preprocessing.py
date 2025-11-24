import pandas as pd
import os
from src.preprocessing import Preprocessing

df = pd.read_csv('data/engineered/engineered_dataset.csv')

preprocessing = Preprocessing(df)
df = preprocessing.encoding().scaling().logTransformation().getDataset()


# save
output_folder = 'data/preprocessed'
os.makedirs(output_folder, exist_ok=True)

output_path = os.path.join(output_folder, 'preprocessed_dataset.csv')
df.to_csv(output_path, index=False)

print(df.head(10))