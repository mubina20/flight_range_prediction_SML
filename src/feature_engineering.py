import pandas as pd

class FeatureCreation:
    def __init__(self, df):
        self.df = df.copy()

    # create company
    def create_Company(self):
        self.df['Company'] = self.df['Model'].astype(str).str.split(' ').str[0]
        return self

    # bir odamga ketodigon HMC narhi
    def create_HMC_per_person(self):
        self.df['HMC_per_person'] = self.df['Hourly_Maintenance_Cost_($)'] / self.df['Capacity']
        return self

    # har km dan narhi
    def create_Cost_per_km(self):
        self.df['Cost_per_km'] = self.df['Price_($)'] / self.df['Range_(km)']
        return self
    
    # dataset 2023ga moslangan bulgan uchun 2025 ga utkazish
    def change_Age(self):
        self.df['Age'] = 2025 - self.df['Year_of_Manufacture']
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