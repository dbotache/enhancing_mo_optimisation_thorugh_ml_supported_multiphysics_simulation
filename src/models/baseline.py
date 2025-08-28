import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import pywt # https://pywavelets.readthedocs.io/en/latest/index.html
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline





# Polynomial Features
class PolynomialFunction(BaseEstimator, TransformerMixin):
    """
    Transformer that generates polynomial features for each input feature,
    suitable for use in scikit-learn pipelines.

    This transformer expands each feature into powers from 0 (bias term) or 1
    up to the specified order, without interaction terms between different features.
    Missing values are not imputed internally and must be handled beforehand.

    Parameters
    ----------
    order : int, default=2
        The maximum polynomial degree for each feature.

    include_bias : bool, default=True
        If True, includes a bias (constant 1) term for each feature.

    Attributes
    ----------
    n_features_in_ : int
        The number of input features seen during fitting.

    Notes
    -----
    - Only applies powers individually to each feature (no cross-feature terms).
    - The `inverse_transform` method will reconstruct the original features
      only exactly if `order=1` or if the transformation is otherwise invertible.
    - Uses NumPy arrays internally; accepts Pandas DataFrames or Series.

    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LinearRegression
    >>> X = pd.DataFrame({"x1": [1, 2], "x2": [3, 4]})
    >>> poly = PolynomialFunction(order=3, include_bias=False)
    >>> poly.transform(X)
    array([[  1.,   1.,   9.,   27.],
           [  2.,   4.,  16.,  64.]])

    >>> pipe = Pipeline([("poly", PolynomialFunction(order=2)), ("lr", LinearRegression())])
    >>> pipe.fit(X, [1, 2])
    Pipeline(...)
    """
    def __init__(self, order=2, include_bias=True):
        self.order = order
        self.include_bias = include_bias

    def _calculate_features(self, X):
        """Internal method to compute polynomial features."""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        features_list = []
        for k in range(X_values.shape[1]):
            start_power = 0 if self.include_bias else 1
            loc_feature = [X_values[:, k] ** j for j in range(start_power, self.order + 1)]
            features_list.append(np.array(loc_feature))

        return np.vstack(features_list).T

    def fit(self, X, y=None):
        """Fit does nothing here since no parameters are learned."""
        self.n_features_in_ = X.shape[1] if not isinstance(X, pd.Series) else 1
        return self

    def transform(self, X):
        """Generate polynomial features."""
        return self._calculate_features(X)

    def inverse_transform(self, X_poly):
        """
        Reconstruct original features from polynomial features.
        Note: Only exact if order=1 (linear) or if the transformation is invertible.
        """
        if self.include_bias:
            start_idx = 1  # skip bias column
        else:
            start_idx = 0

        original_features = []
        for i in range(self.n_features_in_):
            col_idx = i * (self.order + 1 - start_idx) + start_idx
            original_features.append(X_poly[:, col_idx])
        return np.column_stack(original_features)


# Gaussian Basis Functions
class GaussianBasisGenerator(BaseEstimator, TransformerMixin):
    """
    Transformer that generates Gaussian (Radial Basis Function) features
    from each input feature, suitable for use in scikit-learn pipelines.

    This class expands each original feature into a set of Gaussian basis
    functions, where the centers (mu) are determined either by equally spaced
    values or by KMeans clustering. The standard deviation (scale) of each
    Gaussian is computed from the spacing of the centers, with a small epsilon
    to avoid division-by-zero errors. Missing values are imputed internally.

    Parameters
    ----------
    n_centers : int, default=5
        Number of Gaussian centers per feature.

    method : {'equal_spacing', 'kmeans'}, default='equal_spacing'
        Method for determining the Gaussian centers:
        - 'equal_spacing' : equally spaced between min and max of feature.
        - 'kmeans'        : cluster centers from KMeans.

    include_bias : bool, default=True
        If True, includes a bias (constant 1) term as the first column for
        each transformed feature block.

    epsilon : float, default=1e-8
        Minimum allowed standard deviation value to avoid numerical instability.

    impute_strategy : {'mean', 'median'}, default='mean'
        Strategy for imputing missing values before transformation.

    Attributes
    ----------
    mu_ : list of ndarray
        List of arrays containing the Gaussian centers for each feature.

    s_ : list of float
        List of scale (standard deviation) values for each feature.

    Notes
    -----
    - Handles constant features by assigning a single center equal to the constant value.
    - Missing values are imputed column-wise using the specified strategy.
    - Returned features are concatenated for all original features.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LinearRegression
    >>> gb = GaussianBasisGenerator(n_centers=3, method='kmeans')
    >>> pipe = Pipeline([("gb", gb), ("lr", LinearRegression())])
    >>> pipe.fit([[0], [1], [2]], [1, 2, 3])
    Pipeline(...)
    """

    def __init__(self, n_centers=5, method="equal_spacing", include_bias=True, epsilon=1e-8, impute_strategy="mean"):
        self.n_centers = n_centers
        self.method = method
        self.include_bias = include_bias
        self.epsilon = epsilon
        self.impute_strategy = impute_strategy

    def _impute(self, X):
        """Simple in-transformer imputation for NaNs."""
        X = np.array(X, dtype=float)
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                if self.impute_strategy == "mean":
                    fill_val = np.nanmean(col_data)
                elif self.impute_strategy == "median":
                    fill_val = np.nanmedian(col_data)
                else:
                    raise ValueError("Unknown impute_strategy.")
                col_data[mask] = fill_val
                X[:, col] = col_data
        return X

    def fit(self, X, y=None):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X = self._impute(X)

        self.mu_ = []
        self.s_ = []

        for feature in X.T:
            if np.all(feature == feature[0]):  # constant feature
                mu_vals = np.array([feature[0]])
                s_val = 1.0
            elif self.method == "equal_spacing":
                mu_vals = np.linspace(np.min(feature), np.max(feature), self.n_centers)
                s_val = np.mean(np.diff(mu_vals))
            elif self.method == "kmeans":
                unique_vals = np.unique(feature)
                if len(unique_vals) < self.n_centers:
                    mu_vals = np.sort(unique_vals)
                else:
                    km = KMeans(n_clusters=self.n_centers, n_init=10, random_state=0)
                    km.fit(feature.reshape(-1, 1))
                    mu_vals = np.sort(km.cluster_centers_.flatten())
                s_val = np.mean(np.diff(mu_vals)) if len(mu_vals) > 1 else 1.0
            else:
                raise ValueError("Invalid method. Choose 'equal_spacing' or 'kmeans'.")

            s_val = max(abs(s_val), self.epsilon)
            self.mu_.append(mu_vals)
            self.s_.append(s_val)

        return self

    def transform(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X = self._impute(X)

        features = []
        for k, feature in enumerate(X.T):
            phi_k = [np.exp(-0.5 * ((feature - mu_j) / self.s_[k]) ** 2) for mu_j in self.mu_[k]]
            if self.include_bias:
                phi_k.insert(0, np.ones(len(feature)))
            features.append(np.column_stack(phi_k))

        Phi = np.hstack(features)
        return np.nan_to_num(Phi)


'''
class GaussianBasisGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, n_centers=5, method="equal_spacing", include_bias=True):
        """
        Gaussian basis function generator that estimates mu and s from data.

        Parameters
        ----------
        n_centers : int
            Number of Gaussian centers per feature.
        method : str
            Method to choose centers ('equal_spacing' or 'kmeans').
        include_bias : bool
            Whether to include a bias term.
        """
        self.n_centers = n_centers
        self.method = method
        self.include_bias = include_bias

    def fit(self, X, y=None):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        self.mu_ = []
        self.s_ = []

        for feature in X.T:
            if self.method == "equal_spacing":
                mu_vals = np.linspace(np.min(feature), np.max(feature), self.n_centers)
            elif self.method == "kmeans":
                km = KMeans(n_clusters=self.n_centers, n_init=10, random_state=0)
                km.fit(feature.reshape(-1, 1))
                mu_vals = np.sort(km.cluster_centers_.flatten())
            else:
                raise ValueError("Invalid method. Choose 'equal_spacing' or 'kmeans'.")

            # s: spacing between centers (use average spacing if variable)
            if len(mu_vals) > 1:
                s_val = np.mean(np.diff(mu_vals))
            else:
                s_val = 1.0

            self.mu_.append(mu_vals)
            self.s_.append(s_val)

        return self

    def transform(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        features = []

        for k, feature in enumerate(X.T):
            phi_k = [np.exp(- (feature - mu_j) ** 2 / (2 * self.s_[k] ** 2)) for mu_j in self.mu_[k]]
            if self.include_bias:
                phi_k.insert(0, np.ones(len(feature)))
            features.append(np.column_stack(phi_k))

        return np.hstack(features)
'''

# Sigmoid Basis Functions
class SigmoidBasisGenerator(BaseEstimator, TransformerMixin):
    """
    Transformer that generates sigmoid basis function features
    from each input feature, suitable for use in scikit-learn pipelines.

    This class expands each original feature into a set of sigmoidal features,
    where the centers (mu) are determined either by equally spaced values
    or by KMeans clustering. The scale parameter for each sigmoid is computed
    from the spacing of the centers, with a small epsilon to avoid
    division-by-zero errors. Missing values are imputed internally.

    Parameters
    ----------
    n_centers : int, default=5
        Number of sigmoid centers per feature.

    method : {'equal_spacing', 'kmeans'}, default='equal_spacing'
        Method for determining the sigmoid centers:
        - 'equal_spacing' : equally spaced between min and max of feature.
        - 'kmeans'        : cluster centers from KMeans.

    include_bias : bool, default=True
        If True, includes a bias (constant 1) term as the first column for
        each transformed feature block.

    epsilon : float, default=1e-8
        Minimum allowed scale value to avoid numerical instability.

    impute_strategy : {'mean', 'median'}, default='mean'
        Strategy for imputing missing values before transformation.

    Attributes
    ----------
    mu_ : list of ndarray
        List of arrays containing the sigmoid centers for each feature.

    s_ : list of float
        List of scale values for each feature.

    Notes
    -----
    - Handles constant features by assigning a single center equal to the constant value.
    - Missing values are imputed column-wise using the specified strategy.
    - Returned features are concatenated for all original features.

    Examples
    --------
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.linear_model import LinearRegression
    >>> sb = SigmoidBasisGenerator(n_centers=3, method='equal_spacing')
    >>> pipe = Pipeline([("sb", sb), ("lr", LinearRegression())])
    >>> pipe.fit([[0], [1], [2]], [1, 2, 3])
    Pipeline(...)
    """

    def __init__(self, n_centers=5, method="equal_spacing", include_bias=True, epsilon=1e-8, impute_strategy="mean"):
        self.n_centers = n_centers
        self.method = method
        self.include_bias = include_bias
        self.epsilon = epsilon
        self.impute_strategy = impute_strategy

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def _impute(self, X):
        """Simple in-transformer imputation for NaNs."""
        X = np.array(X, dtype=float)
        for col in range(X.shape[1]):
            col_data = X[:, col]
            mask = np.isnan(col_data)
            if mask.any():
                if self.impute_strategy == "mean":
                    fill_val = np.nanmean(col_data)
                elif self.impute_strategy == "median":
                    fill_val = np.nanmedian(col_data)
                else:
                    raise ValueError("Unknown impute_strategy.")
                col_data[mask] = fill_val
                X[:, col] = col_data
        return X

    def fit(self, X, y=None):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X = self._impute(X)

        self.mu_ = []
        self.s_ = []

        for feature in X.T:
            if np.all(feature == feature[0]):  # constant feature
                mu_vals = np.array([feature[0]])
                s_val = 1.0
            elif self.method == "equal_spacing":
                mu_vals = np.linspace(np.min(feature), np.max(feature), self.n_centers)
                s_val = np.mean(np.diff(mu_vals))
            elif self.method == "kmeans":
                unique_vals = np.unique(feature)
                if len(unique_vals) < self.n_centers:
                    mu_vals = np.sort(unique_vals)
                else:
                    km = KMeans(n_clusters=self.n_centers, n_init=10, random_state=0)
                    km.fit(feature.reshape(-1, 1))
                    mu_vals = np.sort(km.cluster_centers_.flatten())
                s_val = np.mean(np.diff(mu_vals)) if len(mu_vals) > 1 else 1.0
            else:
                raise ValueError("Invalid method. Choose 'equal_spacing' or 'kmeans'.")

            s_val = max(abs(s_val), self.epsilon)
            self.mu_.append(mu_vals)
            self.s_.append(s_val)

        return self

    def transform(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        X = self._impute(X)

        features = []
        for k, feature in enumerate(X.T):
            phi_k = [self._logistic((feature - mu_j) / self.s_[k]) for mu_j in self.mu_[k]]
            if self.include_bias:
                phi_k.insert(0, np.ones(len(feature)))
            features.append(np.column_stack(phi_k))

        Phi = np.hstack(features)
        return np.nan_to_num(Phi)


'''
class SigmoidBasisGenerator(BaseEstimator, TransformerMixin):
    def __init__(self, n_centers=5, method="equal_spacing", include_bias=True):
        """
        Sigmoid basis function generator that estimates mu and s from data.

        Parameters
        ----------
        n_centers : int
            Number of sigmoid centers per feature.
        method : str
            Method to choose centers ('equal_spacing' or 'kmeans').
        include_bias : bool
            Whether to include a bias term.
        """
        self.n_centers = n_centers
        self.method = method
        self.include_bias = include_bias

    def _logistic(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, X, y=None):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        self.mu_ = []
        self.s_ = []

        for feature in X.T:
            if self.method == "equal_spacing":
                mu_vals = np.linspace(np.min(feature), np.max(feature), self.n_centers)
            elif self.method == "kmeans":
                km = KMeans(n_clusters=self.n_centers, n_init=10, random_state=0)
                km.fit(feature.reshape(-1, 1))
                mu_vals = np.sort(km.cluster_centers_.flatten())
            else:
                raise ValueError("Invalid method. Choose 'equal_spacing' or 'kmeans'.")

            # s: spacing between centers (use average spacing if variable)
            if len(mu_vals) > 1:
                s_val = np.mean(np.diff(mu_vals))
            else:
                s_val = 1.0

            self.mu_.append(mu_vals)
            self.s_.append(s_val)

        return self

    def transform(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
        features = []

        for k, feature in enumerate(X.T):
            phi_k = [self._logistic((feature - mu_j) / self.s_[k]) for mu_j in self.mu_[k]]
            if self.include_bias:
                phi_k.insert(0, np.ones(len(feature)))
            features.append(np.column_stack(phi_k))

        return np.hstack(features)

        '''

class FourierBasisFunction(BaseEstimator, TransformerMixin):
    """
    Fourier (trigonometric) basis expansion for periodic data.

    Parameters
    ----------
    n_terms : int
        Number of sine/cosine terms (frequency harmonics) to include.
    period : float
        The fundamental period of the data.
    include_bias : bool
        Whether to include a bias term (constant 1).
    """
    def __init__(self, n_terms=3, period=1.0, include_bias=True):
        self.n_terms = n_terms
        self.period = period
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        features = []
        if self.include_bias:
            features.append(np.ones((X_values.shape[0], 1)))

        for j in range(1, self.n_terms + 1):
            features.append(np.sin(2 * np.pi * j * X_values / self.period))
            features.append(np.cos(2 * np.pi * j * X_values / self.period))

        return np.hstack(features)

    def inverse_transform(self, X_basis):
        # Fourier transform is not trivially invertible in this feature form
        raise NotImplementedError("Inverse transform not supported for Fourier basis.")


class WaveletBasisFunction(BaseEstimator, TransformerMixin):
    """
    Wavelet basis expansion using discrete wavelet transform (DWT).

    Parameters
    ----------
    wavelet : str
        Wavelet name (e.g., 'haar', 'db1', 'db2').
    level : int
        Decomposition level.
    """
    def __init__(self, wavelet='haar', level=2):
        self.wavelet = wavelet
        self.level = level

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        all_coeffs = []
        for col in range(X_values.shape[1]):
            coeffs = pywt.wavedec(X_values[:, col], self.wavelet, level=self.level)
            flattened = np.hstack([c for c in coeffs])
            all_coeffs.append(flattened)
        return np.array(all_coeffs).T

    def inverse_transform(self, X_basis):
        raise NotImplementedError("Inverse transform not implemented for wavelet basis.")


class MultiquadricBasisFunction(BaseEstimator, TransformerMixin):
    """
    Multiquadric radial basis function expansion with automatic center estimation.

    Parameters
    ----------
    n_centers : int
        Number of RBF centers to estimate using KMeans.
    c : float, default=1.0
        Shape parameter controlling the width of the basis.
        If 'auto', it will be set to the average distance between centers.
    include_bias : bool, default=True
        Whether to include a bias term.
    random_state : int or None
        Random state for reproducibility in KMeans.
    """

    def __init__(self, n_centers=5, c=1.0, include_bias=True, random_state=None):
        self.n_centers = n_centers
        self.c = c
        self.include_bias = include_bias
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit KMeans to determine centers and set shape parameter if needed."""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        self.kmeans_ = KMeans(
            n_clusters=self.n_centers,
            random_state=self.random_state
        ).fit(X_values)

        self.centers_ = self.kmeans_.cluster_centers_

        if self.c == "auto":
            dists = []
            for i in range(len(self.centers_)):
                for j in range(i + 1, len(self.centers_)):
                    dists.append(np.linalg.norm(self.centers_[i] - self.centers_[j]))
            self.c_ = np.mean(dists)
        else:
            self.c_ = float(self.c)

        self.n_features_in_ = X_values.shape[1]
        return self

    def transform(self, X):
        """Transform input data into multiquadric RBF features."""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        features = []
        if self.include_bias:
            features.append(np.ones((X_values.shape[0], 1)))

        for mu in self.centers_:
            r = np.linalg.norm(X_values - mu, axis=1)
            features.append(np.sqrt((r / self.c_) ** 2 + 1).reshape(-1, 1))  # ensure 2D

        return np.hstack(features)

    def _old_transform(self, X):
        """Transform input data into multiquadric RBF features."""
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        features = []
        if self.include_bias:
            features.append(np.ones((X_values.shape[0], 1)))

        for mu in self.centers_:
            r = np.linalg.norm(X_values - mu, axis=1)
            features.append(np.sqrt((r / self.c_) ** 2 + 1))

        return np.hstack(features)

    def inverse_transform(self, X_basis):
        """Inverse transformation not implemented for RBF basis."""
        raise NotImplementedError("Inverse transform not implemented for RBF basis.")


'''

class MultiquadricBasisFunction(BaseEstimator, TransformerMixin):
    """
    Multiquadric radial basis function expansion.

    Parameters
    ----------
    centers : array-like
        Centers (mu values) for the RBFs.
    c : float
        Shape parameter controlling the width of the basis.
    include_bias : bool
        Whether to include a bias term.
    """
    def __init__(self, centers, c=1.0, include_bias=True):
        self.centers = np.array(centers)
        self.c = c
        self.include_bias = include_bias

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = np.array(X)

        features = []
        if self.include_bias:
            features.append(np.ones((X_values.shape[0], 1)))

        for mu in self.centers:
            r = np.linalg.norm(X_values - mu, axis=1)
            features.append(np.sqrt((r / self.c) ** 2 + 1))

        return np.hstack(features)

    def inverse_transform(self, X_basis):
        raise NotImplementedError("Inverse transform not implemented for RBF basis.")

'''


class MultiOutputSVR(BaseEstimator, RegressorMixin):
    """
    MultiOutputSVR(kernel='rbf', C=1.0, epsilon=0.1, gamma='scale', degree=3, coef0=0.0)

    A multi-output regression wrapper for scikit-learn's Support Vector Regressor (SVR).

    The standard `sklearn.svm.SVR` implementation only supports **single-target regression**
    (i.e., predicting a single continuous output variable). This class extends SVR to
    handle **multi-output regression** by fitting one independent SVR model per output dimension.

    This implementation is compatible with the scikit-learn API, so it can be:
        - Used directly in `Pipeline` objects
        - Tuned with `GridSearchCV` or `RandomizedSearchCV`
        - Cloned and serialized like any other scikit-learn estimator

    Parameters
    ----------
    kernel : str, default='rbf'
        Specifies the kernel type to be used in the SVR model.
        Must be one of: {'linear', 'poly', 'rbf', 'sigmoid', 'precomputed'}.

    C : float, default=1.0
        Regularization parameter. The strength of the regularization is inversely
        proportional to C. Must be strictly positive.

    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model. It specifies the margin of tolerance
        where no penalty is given to errors.

    gamma : {'scale', 'auto'} or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels.
        - If 'scale', uses 1 / (n_features * X.var()) as the value of gamma.
        - If 'auto', uses 1 / n_features.

    degree : int, default=3
        Degree of the polynomial kernel function ('poly'). Ignored for other kernels.

    coef0 : float, default=0.0
        Independent term in kernel function. Used only in 'poly' and 'sigmoid'.

    Attributes
    ----------
    estimators_ : list of sklearn.svm.SVR objects
        List containing one fitted SVR model for each output dimension.

    n_outputs_ : int
        Number of output dimensions in the training data.

    n_features_in_ : int
        Number of features in the input data.

    Methods
    -------
    fit(X, y):
        Fit one SVR model per output dimension.

    predict(X):
        Predict outputs for the given input data by stacking the predictions from each SVR.

    Notes
    -----
    - Each target dimension is treated independently — there is no interaction or shared kernel
      optimization between outputs.
    - Fitting multiple SVRs may be computationally expensive for large datasets or many outputs.
    - Can be combined with feature transformation steps in a scikit-learn Pipeline for preprocessing.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split
    >>> X, y = make_regression(n_samples=100, n_features=5, n_targets=3, noise=0.1)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    >>> model = MultiOutputSVR(kernel='rbf', C=10, epsilon=0.01)
    >>> model.fit(X_train, y_train)
    MultiOutputSVR(C=10, epsilon=0.01)

    >>> preds = model.predict(X_test)
    >>> preds.shape
    (25, 3)  # 25 samples, 3 target dimensions
    """
    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale", degree=3, coef0=0.0):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

    def fit(self, X, y):
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.estimators_ = []
        for i in range(y.shape[1]):
            svr = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon,
                      gamma=self.gamma, degree=self.degree, coef0=self.coef0)
            svr.fit(X, y[:, i])
            self.estimators_.append(svr)
        return self

    def predict(self, X):
        preds = [est.predict(X).reshape(-1, 1) for est in self.estimators_]
        return np.hstack(preds)


