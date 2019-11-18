import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.metrics import euclidean_distances
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y
from sklearn.utils.estimator_checks import check_estimator


class MyScaler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # y is ignored
        X = check_array(X)
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X):
        X = check_array(X)
        return (X - self.mean_) / self.std_


class MyNeighborsClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y
        return self

    def predict(self, X):
        check_is_fitted(self, ["X_", "y_"])
        X = check_array(X)
        closest = np.argmin(euclidean_distances(X, self.X_), axis=1)
        return self.y_[closest]


check_estimator(MyScaler)
check_estimator(MyNeighborsClassifier)
