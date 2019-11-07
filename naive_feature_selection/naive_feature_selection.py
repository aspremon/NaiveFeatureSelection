import numpy as np

from typing import Union

from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from naive_feature_selection.sparse_naive_bayes import sparse_naive_bayes


class NaiveFeatureSelection(BaseEstimator, SelectorMixin):
    """Naive feature selection.
    Parameters
    ----------
    k : number of variables to select.
    alpha : typo robustness parameter (same alpha as in MNB from sklearn)
    """

    def __init__(self, k: Union[int, str], alpha: float = 1e-10):
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
