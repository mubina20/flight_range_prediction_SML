import pandas as pd
import os
from joblib import load, dump
from src.preprocessing import Preprocessing
from src.logger import get_logger

logger = get_logger('use_preprocessing', 'preprocessing.log')

df = pd.read_csv('data/engineered/engineered_dataset.csv')

preprocessing = Preprocessing(df, target='Range_(km)')
df_preprocessed = preprocessing.encode(include_targets=True).scale().get_dataset()
logger.info("Preprocessing steps completed successfully.")

os.makedirs('data/preprocessed', exist_ok=True)
df_preprocessed.to_csv('data/preprocessed/preprocessed_dataset.csv', index=False)
logger.info("Preprocessed dataset saved to data/preprocessed/preprocessed_dataset.csv")

dump(preprocessing, 'pipeline/preprocessed_pipeline.joblib')
logger.info("Preprocessing pipeline saved to pipeline/preprocessed_pipeline.joblib")

logger.info("Successfully preprocessed the dataset and saved all outputs!")