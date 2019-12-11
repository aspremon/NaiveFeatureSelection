import numpy as np

from typing import Union

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from naive_feature_selection.sparse_naive_bayes import (
    sparse_naive_bayes,
    sparse_naive_bayes_from_averages,
    fast_naive_bayes_data_averages,
)


class NaiveFeatureSelection(BaseEstimator, SelectorMixin):
    """Naive feature selection.
    Parameters
    ----------
    k : number of variables to select.
    alpha : typo robustness parameter (same alpha as in MNB from sklearn)
    """

    def __init__(self, k: Union[int, str] = 'all', alpha: float = 1e-10):
        self.k = k
        self.alpha = alpha

    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                f"k should be >= 0, <= n_features = {X.shape[1]}; "
                f"got {self.k}. Use k='all' to return all features."
            )

    def fit(self, X, y):
        """Run naive feature selection on (X, y) with target k and get 
        the appropriate features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        (k : target number of variables)
        Returns
        -------
        self : object
        """
        # Check params
        X, y = check_X_y(X, y, ["csr", "csc"], multi_output=True)
        self._check_params(X, y)
        # Get features
        if self.k == "all":
            mask = np.ones(X.shape[1], dtype=bool)
        elif self.k == 0:
            mask = np.zeros(X.shape[1], dtype=bool)
        else:
            res_nfs = sparse_naive_bayes(X, y, self.k, alpha=self.alpha)
            mask = np.zeros(X.shape[1], dtype=bool)
            mask[res_nfs["idx"]] = 1

        self.mask_ = mask
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "mask_")
        return self.mask_


class StreamedNaiveFeatureSelection(NaiveFeatureSelection):

    def __init__(self, k: Union[int, str], num_features: int = None, alpha: float = 1e-10):
        super().__init__(k, alpha=alpha)
        if num_features is None:
            self.f1 = None
            self.f2 = None
        else:
            self.f1 = np.zeros(num_features)
            self.f2 = np.zeros(num_features)
        self.num_features = num_features
        self.num_class1 = 0
        self.num_total = 0

    def partial_fit(self, x, y):
        x, y = check_X_y(x, y, ['csr', 'csc'], multi_output=True)
        self._check_params(x, y)

        if self.num_features is None:
            self.num_features = x.shape[1]
            self.f1 = np.zeros(self.num_features)
            self.f2 = np.zeros(self.num_features)

        f1, f2, pc1, pc2 = fast_naive_bayes_data_averages(x, y, alpha=0)

        self.f1 += f1
        self.f2 += f2

        self.num_class1 += pc1 * x.shape[0]
        self.num_total += x.shape[0]

    def fit(self, x: np.ndarray = None, y: np.ndarray = None):
        if x is not None and y is not None:
            self.partial_fit(x, y)

        if self.k == 'all':
            mask = np.ones(self.num_features, dtype=bool)
        elif self.k == 0:
            mask = np.zeros(self.num_features, dtype=bool)
        else:
            pc1 = self.num_class1 / self.num_total
            pc2 = 1 - pc1
            f1 = self.f1 + self.alpha * self.num_class1
            f2 = self.f2 + self.alpha * (self.num_total - self.num_class1)
            res_nfs = sparse_naive_bayes_from_averages(f1, f2, pc1, pc2, self.k)
            mask = np.zeros(self.num_features, dtype=bool)
            mask[res_nfs['idx']] = 1

        self.mask_ = mask
        return self
