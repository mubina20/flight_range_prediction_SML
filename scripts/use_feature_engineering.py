import pandas as pd
import os
from src.feature_engineering import FeatureCreatorTransformer
from joblib import dump
from src.logger import get_logger

logger = get_logger('use_feature_engineering', 'feature_engineering.log')
df = pd.read_csv('data/raw/airplane_dataset.csv')

feature_creator = FeatureCreatorTransformer()
df_features = feature_creator.transform(df)
logger.info("Engineering steps completed successfully.")

os.makedirs('data/engineered', exist_ok=True)
df_features.to_csv('data/engineered/engineered_dataset.csv', index=False)
logger.info("Engineered dataset saved to data/engineered/engineered_dataset.csv")

dump(feature_creator, 'pipeline/feature_pipeline.joblib')
logger.info("Engineering pipeline saved to pipeline/feature_pipeline.joblib")

logger.info("Successfully engineered the dataset and saved all outputs!")