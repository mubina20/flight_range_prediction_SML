import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.logger import get_logger

logger = get_logger('feature_engineering', 'feature_engineering.log')

class FeatureCreatorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X['Company'] = X['Model'].str.split(' ').str[0]
        X['Age'] = 2025 - X['Year_of_Manufacture']
        X['Age_Group'] = pd.cut(
            X['Age'],
            bins=[0, 10, 20, 30, 40, 200],
            labels=['0-10','10-20','20-30','30-40','40+']
        )

        X['HMC_per_person'] = (
            X['Hourly_Maintenance_Cost_($)'] /
            X['Capacity']
        )

        engine_power_map = {
            'Turbofan': 6,
            'Piston': 1,
            'Turboprop': 3
        }
        X['Engine_Power_Factor'] = (
            X['Number_of_Engines'] *
            X['Engine_Type'].map(engine_power_map)
        )

        X['Price_per_Seat'] = (
            X['Price_($)'] /
            X['Capacity']
        )

        X['Seats_per_Engine'] = (
            X['Capacity'] /
            X['Number_of_Engines']
        )

        company_counts = X['Company'].value_counts()
        X['Company_Popularity'] = X['Company'].map(company_counts)

        X['FuelCost_Maint_Index'] = (
            X['Fuel_Consumption_(L/hour)'] * 1.0 +
            X['Hourly_Maintenance_Cost_($)'] * 0.01
        )

        X['Engine_to_Capacity'] = (
            X['Number_of_Engines'] /
            X['Capacity']
        )

        logger.info("New engineered features successfully added.")

        return X

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'logger' in state:
            del state['logger']
        return state

# class FeatureSelection:
#     def __init__(self, df, target):
#         self.df = df.copy()
#         self.target = target
    
#     def filter_by_correlation(self):
#         corr = self.df.corr()[self.target].abs()
#         selected_features = corr[corr >= 0.2].index.tolist()

#         if self.target in selected_features:
#             selected_features.remove(self.target)

#         self.selected_features = selected_features

#         return self
    
#     def get_selected_features(self):
#         return self.selected_features