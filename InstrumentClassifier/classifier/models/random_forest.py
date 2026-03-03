import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


class RandomForestStrategy:

    def __init__(self, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        if X_train.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_train.shape}.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("X and y must have same number of samples.")

        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def get_name(self):
        return "RandomForest"
