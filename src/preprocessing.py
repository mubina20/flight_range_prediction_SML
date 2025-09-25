import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np

df = pd.read_csv("data/raw/airplane_price_dataset.csv")
encoder = LabelEncoder()
scaler = MinMaxScaler()

# Preprocessing class
class DataPreprocessing:
    def __init__(self, df):
        self.df = df.copy()

    def fillMissingValues(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == 'object':
                    self.df[col].fillna(self.df[col].mode()[0], inplace=True)
                else:
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
        return self
    
    def encoding(self):
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                if self.df[col].nunique() <= 5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat([self.df.drop(columns=col), dummies], axis=1)
                else:
                    self.df[col]=encoder.fit_transform(self.df[col])
        return self
    
    def scaling(self):
        num_col = self.df.select_dtypes(include=['int64', 'float64']).columns.drop('Range_(km)')
        self.df[num_col] = scaler.fit_transform(self.df[num_col])
        return self
    
    def logTransformation(self):
        skewness = self.df.skew()
        feature_log = skewness[(skewness >= 0.5)].index.tolist()

        for col in feature_log:
            if (self.df[col] > 0).all():
                self.df[col] = np.log1p(self.df[col])

        return self
    
    def getDataset(self):
        return self.df