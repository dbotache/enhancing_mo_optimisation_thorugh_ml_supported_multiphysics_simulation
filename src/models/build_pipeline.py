from .baseline import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.linear_model import ElasticNet, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# --- Default Features ---
features_defaults = {
    "Polynomial": {"features__order": 2},  # quadratic
    "Wavelets": {"features__wavelet": "haar"},
    "Fourier": {"features__n_terms": 10},
    "Sigmoid": {
        "features__n_centers": 10,
        "features__method": "equal_spacing"
    },
    "Gaussian": {
        "features__n_centers": 10,
        "features__method": "equal_spacing"
    },
    "RBF": {"features__n_centers": 10}
}

# --- Default Dimensionality Reduction ---
dimred_defaults = {
    "PCA": {"dimred__n_components": 100},
    "NMF": {
        "dimred__n_components": 5,
        "dimred__init": "nndsvd",
        "dimred__solver": "cd",
        "dimred__alpha_W": 0.0,
        "dimred__l1_ratio": 0.0
    }
}

# --- Default Models ---
models_defaults = {
    "Elastic Net": {"model__alpha": 0.01, "model__l1_ratio": 0.5},
    "Ridge Regression": {"model__alpha": 1.0},
    "Lasso Regression": {"model__alpha": 0.01},
    "Decision Tree": {"model__max_depth": 5, "model__min_samples_split": 2},
    "Random Forest": {"model__n_estimators": 100, "model__max_depth": 10},
    "Support Vector Regressor": {"model__C": 1, "model__kernel": "rbf"}
}


# --- Grid Parameters --- Features
features_dict = {
    "Polynomial": (PolynomialFunction(), {"order": [1, 2, 3, 4, 5, 6, 7]}),
    "Wavelets": (WaveletBasisFunction(), {"wavelet": ['haar', 'db1', 'db2']}),
    "Fourier": (FourierBasisFunction(), {"n_terms": np.arange(2,150)}),
    "Sigmoid": (SigmoidBasisGenerator(),
               {
                   "n_centers": np.arange(2, 150),
                   "method": ["equal_spacing", "kmeans"]
               }),
    "Gaussian": (GaussianBasisGenerator(),
                 {
                     "n_centers": np.arange(2, 150),
                     "method": ["equal_spacing", "kmeans"],
                 }),
    "RBF": (MultiquadricBasisFunction(), {"n_centers": np.arange(2, 150)})
}

# --- Grid Parameters --- dimensionality reduction
dimension_reduction = {
    "PCA": (PCA(), {"n_components": [100, 500, 1000]}),
    "NMF": (NMF(),
            {
                "n_components": [5, 10, 20],
                "init": ["nndsvd", "random"],
                "solver": ["cd", "mu"],
                "alpha_W": [0.0, 0.1],
                "l1_ratio": [0.0, 0.5, 1.0]
            })
}

# --- Grid Parameters --- Models Hyperparameters
models_dict = {
    "Elastic Net": (ElasticNet(),
                    {
                        "alpha": [0.001, 0.01, 0.1, 1.0],
                        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
                    }),
    "Ridge Regression": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
    "Lasso Regression": (Lasso(), {"alpha": [0.001, 0.01, 0.1, 1.0]}),
    "Decision Tree": (DecisionTreeRegressor(), {
        "max_depth": [3, 5, 10, None],
        "min_samples_split": [2, 5, 10]
    }),
    "Random Forest": (RandomForestRegressor(), {
        "n_estimators": [50, 100],
        "max_depth": [5, 10, None]
    }),
    "Support Vector Regressor": (MultiOutputSVR(), {
        "C": [0.1, 1, 10],
        "kernel": ["linear", "rbf"]
    })
}


class AutoPipeline:
    def __init__(self, features_dict, dimred_dict, models_dict):
        self.features_dict = features_dict
        self.dimred_dict = dimred_dict
        self.models_dict = models_dict
        self.last_grid = None  # store last fitted GridSearchCV
        self.last_estimator = None  # store last manually trained pipeline

        self.scalers = {
            "None": ("passthrough", {}),
            "StandardScaler": (StandardScaler(), {}),
            "MinMaxScaler": (MinMaxScaler(), {})
        }

    def build_pipeline(self, scaler_name=None, feature_name=None, dimred_name=None, model_name=None):
        """
        Build a pipeline dynamically. Any component can be set to None to skip it.
        """
        steps = []
        param_grid = {}

        # --- Feature Transformation ---
        if feature_name is not None and feature_name != "None":
            feature, feat_params = self.features_dict[feature_name]
            steps.append(("features", feature))
            param_grid.update({f"features__{k}": v for k, v in feat_params.items()})

        # --- Handle Scaler (AFTER features if present) ---
        if scaler_name is not None and scaler_name != "None":
            scaler, _ = self.scalers[scaler_name]
            steps.append(("scaler", scaler))

        # --- Dimensionality Reduction ---
        if dimred_name is not None and dimred_name != "None":
            dimred, dim_params = self.dimred_dict[dimred_name]
            steps.append(("dimred", dimred))
            param_grid.update({f"dimred__{k}": v for k, v in dim_params.items()})

        # --- Model ---
        if model_name is not None and model_name != "None":
            model, model_params = self.models_dict[model_name]
            steps.append(("model", model))
            param_grid.update({f"model__{k}": v for k, v in model_params.items()})

        if not steps:
            raise ValueError("Pipeline must contain at least one component (features, scaler, dimred, or model).")

        pipeline = Pipeline(steps)
        return pipeline, param_grid

    def run_gridsearch(self, X, y, scaler_name, feature_name, dimred_name, model_name,
                       cv=5, scoring="neg_mean_squared_error", n_jobs=-1):
        pipeline, param_grid = self.build_pipeline(scaler_name, feature_name, dimred_name, model_name)

        grid = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1)
        grid.fit(X, y)

        self.last_grid = grid
        self.last_estimator = grid.best_estimator_
        return grid

    def train_pipeline(self, X, y, scaler_name, feature_name, dimred_name, model_name, params: dict = None):
        """Train a pipeline directly with user-provided params (no GridSearch)."""
        pipeline, _ = self.build_pipeline(scaler_name, feature_name, dimred_name, model_name)

        if params is not None:
            pipeline.set_params(**params)

        pipeline.fit(X, y)
        self.last_estimator = pipeline
        return pipeline

    def predict(self, X):
        if self.last_estimator is None:
            raise ValueError("No fitted pipeline found. Run run_gridsearch() or train_pipeline() first.")
        return self.last_estimator.predict(X)
