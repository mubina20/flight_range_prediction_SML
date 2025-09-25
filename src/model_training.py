from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error 
import os
from joblib import dump

class Trainer:
    def __init__(self, model, x, y):
        self.model = model
        self.x = x
        self.y = y
    
    def train(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2, random_state=42
        )
        self.model.fit(self.x_train, self.y_train)
        return self
    
    def evaluate(self):
        y_pred = self.model.predict(self.x_test)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        self.r2 = r2_score(self.y_test, y_pred)
        self.mae = mean_absolute_error(self.y_test, y_pred)
        self.score = cross_val_score(self.model, self.x, self.y, cv=kf, scoring='r2')
        return self.r2, self.mae, self.score
    
    def save_model(self):
        os.makedirs('models', exist_ok=True)
        dump(self.model, os.path.join('models', f"{self.model.__class__.__name__}.joblib"))