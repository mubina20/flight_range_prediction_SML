import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from src.logger import get_logger

logger = get_logger('preprocessing', 'preprocessing.log')


class Preprocessing(BaseEstimator, TransformerMixin):
    def __init__(self, df: pd.DataFrame = None, target=None, log_transform=True):
        self.df = df.copy() if df is not None else None
        self.target = target if isinstance(target, list) else [target] if target else []
        self.log_transform = log_transform
        
        self.imputers = {}
        self.label_encoders = {}
        self.onehot_columns = []
        self.scalers = {}
        self.log_cols = []

    def encode(self, include_targets=False):
        for col in self.df.columns:
            if not include_targets and col in self.target:
                continue

            if self.df[col].dtype == "object":
                if self.df[col].nunique() <= 5:
                    self.onehot_columns.append(col)
                    logger.info(f"Applied One-Hot Encoding to column: {col}")
                else:
                    enc = LabelEncoder()
                    self.df[col] = enc.fit_transform(self.df[col])
                    self.label_encoders[col] = enc
                    logger.info(f"Applied Label Encoding to column: {col}")

        for col in self.onehot_columns:
            dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
            self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
            logger.info(f"Generated dummy variables for: {col}")

        logger.info("Categorical features successfully encoded.")

        return self
    
    def logTransformation(self):
        if not self.log_transform:
            return self

        skewness = self.df.select_dtypes(include=['int64', 'float64']).skew()
        self.log_cols = skewness[skewness >= 0.5].index.tolist()

        for col in self.log_cols:
            if (self.df[col] > 0).all():
                self.df[col] = np.log1p(self.df[col])

        logger.info(f"Log transformation applied to: {self.log_cols}")

        return self

    def scale(self):

        numeric_cols = self.df.select_dtypes(include=['int64', 'float64']).columns

        for col in numeric_cols:
            if col in self.target:
                continue

            scaler = MinMaxScaler()
            self.df[col] = scaler.fit_transform(self.df[[col]])
            self.scalers[col] = scaler

        logger.info(f"Numerical scaling applied to {len(numeric_cols)} columns using MinMaxScaler().")

        return self

    def transform_new(self, new_df: pd.DataFrame):
        df_copy = new_df.copy()

        # label encoding
        for col, enc in self.label_encoders.items():
            if col in df_copy.columns:
                df_copy[col] = enc.transform(df_copy[col])
                logger.info(f"Applied Label Encoding to column: {col}")

        # one-hot
        for col in self.onehot_columns:
            if col in df_copy.columns:
                dummies = pd.get_dummies(df_copy[col], prefix=col, dtype=int)
                df_copy = pd.concat([df_copy.drop(columns=col), dummies], axis=1)
                logger.info(f"Generated One-Hot dummy variables for: {col}")

        # log transform
        for col in self.log_cols:
            if col in df_copy.columns and (df_copy[col] > 0).all():
                df_copy[col] = np.log1p(df_copy[col])
                logger.info(f"Applied log1p transformation to: {col}")

        # scaling
        for col, scaler in self.scalers.items():
            if col in df_copy.columns:
                df_copy[col] = scaler.transform(df_copy[[col]])
                logger.info(f"Scaled column: {col}")

        return df_copy

    # ---------------------------
    def get_dataset(self):
        return self.df

    # ---------------------------
    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        if 'df' in state:
            del state['df']

        return state

    # ---------------------------
    def fit(self, X, y=None):
        self.df = X.copy()

        self.encode(include_targets=True)
        self.logTransformation()
        self.scale()

        return self

    # ---------------------------
    def transform(self, X):
        return self.transform_new(X)
