import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class FeatureTargetScaling:
    def __init__(self, X, y, scaler_type_features, scaler_type_targets=None):
        self.X = X
        self.y = y

        self.X_scaled = pd.DataFrame()
        self.y_scaled = pd.DataFrame()

        self.feature_scaler = None
        self.target_scaler = None

        if scaler_type_features == "standard":
            self.feature_scaler = StandardScaler()
        elif scaler_type_features == "minmax":
            self.feature_scaler = MinMaxScaler()
        else:
            self.feature_scaler = None

        if scaler_type_targets == "standard":
            self.target_scaler = StandardScaler()
        elif scaler_type_targets == "minmax":
            self.target_scaler = MinMaxScaler()
        else:
            self.target_scaler = None

        if self.feature_scaler:
            X_scaled_ = self.feature_scaler.fit_transform(X)
            self.X_scaled = pd.DataFrame(X_scaled_, columns=X.columns.values, index=X.index)

        else:
            self.X_scaled = X

        if self.target_scaler:
            y_scaled_ = self.target_scaler.fit_transform(y)
            self.y_scaled = pd.DataFrame(y_scaled_, columns=y.columns.values, index=y.index)
        else:
            self.y_scaled = y

    def scale_data(self, X, y=None):
        if self.feature_scaler is not None:
            X_ = self.feature_scaler.fit_transform(X)
            X = pd.DataFrame(X_, columns=X.columns.values, index=X.index)

        if self.target_scaler is not None and y is not None:
            y_ = self.target_scaler.fit_transform(y)
            y = pd.DataFrame(y_, columns=y.columns.values, index=y.index)

        if y is None:
            return X
        else:
            return X, y

    def inverse_transform_targets(self, y):
        if self.target_scaler is not None:
            y_ = self.target_scaler.inverse_transform(y)
            y = pd.DataFrame(y_, columns=y.columns.values, index=y.index)

        return y

    def inverse_transform_features(self, X):
        if self.feature_scaler is not None:
            X_ = self.feature_scaler.inverse_transform(X)
            X = pd.DataFrame(X_, columns=X.columns.values, index=X.index)

        return X
