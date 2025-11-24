import pandas as pd

class FeatureCreation:
    def __init__(self, df):
        self.df = df.copy()

    def create_features(self):

        self.df['Company'] = self.df['Model'].str.split(' ').str[0]

        self.df['Age'] = 2025 - self.df['Year_of_Manufacture']

        self.df['Age_Group'] = pd.cut(
            self.df['Age'],
            bins=[0, 10, 20, 30, 40, 200],
            labels=['0-10','10-20','20-30','30-40','40+']
        )

        self.df['HMC_per_person'] = (
            self.df['Hourly_Maintenance_Cost_($)'] /
            self.df['Capacity']
        )

        engine_power_map = {
            'Turbofan': 6,
            'Piston': 1,
            'Turboprop': 3
        }
        self.df['Engine_Power_Factor'] = (
            self.df['Number_of_Engines'] *
            self.df['Engine_Type'].map(engine_power_map)
        )

        self.df['Price_per_Seat'] = (
            self.df['Price_($)'] /
            self.df['Capacity']
        )

        self.df['Seats_per_Engine'] = (
            self.df['Capacity'] /
            self.df['Number_of_Engines']
        )

        company_counts = self.df['Company'].value_counts()
        self.df['Company_Popularity'] = self.df['Company'].map(company_counts)

        self.df['FuelCost_Maint_Index'] = (
            self.df['Fuel_Consumption_(L/hour)'] * 1.0 +
            self.df['Hourly_Maintenance_Cost_($)'] * 0.01
        )

        self.df['Engine_to_Capacity'] = (
            self.df['Number_of_Engines'] /
            self.df['Capacity']
        )

        return self

    def getDataset(self):
        return self.df

class FeatureSelection:
    def __init__(self, df, target):
        self.df = df.copy()
        self.target = target
    
    def filter_by_correlation(self):
        corr = self.df.corr()[self.target].abs()
        selected_features = corr[corr >= 0.2].index.tolist()

        if self.target in selected_features:
            selected_features.remove(self.target)

        self.selected_features = selected_features

        return self
    
    def get_selected_features(self):
        return self.selected_features