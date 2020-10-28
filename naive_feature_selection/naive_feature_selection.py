import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted
from scipy.sparse import issparse


######################################################################
# Base classes
###################################################################### 


class NaiveFeatureSelection(BaseEstimator, SelectorMixin):
    """Naive feature selection.
    Parameters
    ----------
    k : number of variables to select.
    alpha : typo robustness parameter (same alpha as in MNB from sklearn)
    """

    def __init__(self, k='all', alpha=1e-10):
        self.k = k
        self.alpha = alpha

    def _split_classes(self, X, y, label=0): # TODO: Use scikit learn here
        """
        split_classes function

        Description: Function that takes the original
        data and partitions it into two classes.

        Arguments:
            - X: training data as a n x m matrix (n datapoints
            with m features per data point)
            - Y: class labels for the n different points
            - label: desired class

        Returns:
            - x1: all the data for class = label
            - x2: the data for class != label

        Note: this only works for binary classification
        """
        idx1 = np.where(y == label)[0]
        idx2 = np.where(y != label)[0]
        return {"class1": X[idx1, :], "class2": X[idx2, :]}

    def _class_prob(self, y, label=0): # TODO: Use scikit learn here
        """
        class_prob function

        Description: Function that finds the ratio
        of class1 and class2 labels to use as the
        prior probability for naive bayes classification.

        Arguments:
            - Y: class labels for the n different points
            - label: desired class
        
        Returns:
            - pc1: probability of class = label
            - pc2: probability of class != label
        """
        n = len(y)
        idx = np.where(y == label)[0]
        return {"pc1": len(idx) * 1.0 / n, "pc2": (n - len(idx)) * 1.0 / n}

    def _entr(self, x):
        """
        Negative entropy function.

        This returns sum_i x_i * log(x_i) 
        """
        res = np.zeros(x.shape)
        idx = np.where(x >= 1e-8) #more robust than using np.nonzero(x)
        res[idx] = x[idx] * np.log(x[idx])
        return res

    def _fun(self, a, k, c, f1, f2):
        """
        Auxialiary gradient function.
        """
        h = c - f1 * np.log(a) - f2 * np.log(1 - a)
        idx = np.argpartition(h, -k)[-k:]
        # fval = sum(h[idx])
        nabla_a = -f1 / a + f2 / (1 - a)
        df = np.sum(nabla_a[idx])
        return df


    def _naive_feature_selection(self, X, y, k):
        """
        Naive Feature Selection

        Description: Function that solves the following multinomial naive Bayes
        training problem, written

        max_{q,r}   fp^T log q + fm^T log r

            s.t. 	\|q - r\|_0 <= k
                    0 <= q, r
                    sum(q) = 1, sum(r) = 1

        Arguments:
            - X: training data as a n x m matrix (n datapoints
            with m features per data point)
            - Y: binary class labels for the n different points (one of the labels should be 1)
            - k: target cardinality for vector in linear classification rule

        Returns:
            - idx: 	Index of selected features
            - q,r: 	Solutions reconstructed from optimal dual solution
            - w0,w: linear classification rule: w0 + w'*x > 0, where w is k sparse.
        """

        # First construct fp, fm, by summing feature vectors for each class
        split = self._split_classes(X, y, label=1)

        C1 = split['class1'].shape[0]
        C2 = split['class2'].shape[0]

        f1 = np.sum(split["class1"], axis=0)
        f2 = np.sum(split["class2"], axis=0)
        f1 = np.squeeze(np.asarray(f1))
        f2 = np.squeeze(np.asarray(f2))

        f1 += self.alpha*C1*np.ones(f1.shape)
        f2 += self.alpha*C2*np.ones(f2.shape)

        # Define dual objective function
        alpha_low = 0
        alpha_top = 1
        c = self._entr(f1) + self._entr(f2) - self._entr(f1 + f2)
        # Solve dual problem by bisection
        tol = 1e-6
        while (alpha_top - alpha_low) > 1e-10:
            alpha = (alpha_top + alpha_low) / 2.0
            df = self._fun(alpha, k, c, f1, f2)
            # print(df)
            if df > tol:
                alpha_top = alpha
            elif df < -tol:  # TODO: check if OK stopping on gradient?
                alpha_low = alpha
            else:
                break

        # Get primal points from dual solution
        h = c - f1 * np.log(alpha) - f2 * np.log(1 - alpha)
        idx = np.argpartition(h, -k)[-k:]
        mask = np.ones(len(f1), dtype=bool)  # all elements included/True.
        mask[idx] = False
        qopt = np.zeros(len(f1))
        ropt = np.zeros(len(f2))
        qopt[mask] = (f1[mask] + f2[mask]) / sum(f1 + f2)
        ropt[mask] = qopt[mask]
        qopt[idx] = sum(f1[idx] + f2[idx]) / (sum(f1[idx]) * sum(f1 + f2)) * f1[idx]
        ropt[idx] = sum(f1[idx] + f2[idx]) / (sum(f2[idx]) * sum(f1 + f2)) * f2[idx]
        objv = f1[np.nonzero(f1)].dot(np.log(qopt[np.nonzero(f1)])) + f2[
            np.nonzero(f2)
        ].dot(np.log(ropt[np.nonzero(f2)]))

        # Get linear classification rule: w0 + w'*x > 0
        pc = self._class_prob(y, label=1)
        w0 = np.log(pc["pc1"]) - np.log(pc["pc2"])
        with np.errstate(divide="ignore"):
            w1 = np.log(qopt)
            idx_inf = np.isinf(w1)
            w1[idx_inf] = -33.6648265  # essentially the 0 entries, log(0) to -33.6648
            w2 = np.log(ropt)
            idx_inf = np.isinf(w2)
            w2[idx_inf] = -33.6648265
        w = w1 - w2

        # Return results
        return {"idx": idx, "w0": w0, "w": w, "q": qopt, "r": ropt, "objv": objv}


    def _binary_naive_feature_selection(self, X, y, k):
        """
        Binary Naive Feature Selection

        Description: Function that solves the following bernoulli naive Bayes
        training problem, written

        max_{q,r}   fp^T log q + fm^T log r

            s.t.    \|q - r\|_0 <= k
                    0 <= q, r <=1
                   

        Arguments:
            - X: training data as a n x m matrix (n datapoints
            with m features per data point)
            - Y: binary class labels for the n different points (one of the labels should be 1)
            - k: target cardinality for vector in linear classification rule

        Returns:
            - idx:  Index of selected features
            - q,r:  Solutions reconstructed from optimal dual solution
            - w0,w: linear classification rule: w0 + w'*x > 0, where w is k sparse.
        """

        # First construct fp, fm, by summing feature vectors for each class
        split = self._split_classes(X, y, label=1)

        C1 = split['class1'].shape[0]
        C2 = split['class2'].shape[0]
        

        f1 = np.sum(split["class1"], axis=0)
        f2 = np.sum(split["class2"], axis=0)
        f1 = np.squeeze(np.asarray(f1))
        f2 = np.squeeze(np.asarray(f2))

        f1 += self.alpha*C1*np.ones(f1.shape)
        f2 += self.alpha*C2*np.ones(f2.shape)

        #laplace smoothing is like missing counts, so we need to add them
        #to our total counts in C1 and C2 respectively
        C1 += self.alpha  
        C2 += self.alpha
        n = C1 + C2

        v = self._entr(f1+f2) - (f1+f2)*np.log(n) + self._entr(n-f1-f2) - (n-f1-f2)*np.log(n)
        wp = self._entr(f1) - f1*np.log(C1) + self._entr(C1-f1) - f1*np.log(C1)
        wm = self._entr(f2) - f2*np.log(C2) + self._entr(C2-f2) - f2*np.log(C2)


        idx = np.argpartition(wp+wm-v, -k)[-k:]
        mask = np.ones(len(f1), dtype=bool)  # all elements included/True.
        mask[idx] = False
        qopt = np.zeros(len(f1))
        ropt = np.zeros(len(f2))
        qopt[mask] = (f1[mask] + f2[mask]) / (C1 + C2)
        ropt[mask] = qopt[mask]
        qopt[idx] = f1[idx]/C1
        ropt[idx] = f2[idx]/C2
        objv = f1[np.nonzero(f1)].dot(np.log(qopt[np.nonzero(f1)])) + f2[
            np.nonzero(f2)
        ].dot(np.log(ropt[np.nonzero(f2)]))

        # Get linear classification rule: w0 + w'*x > 0
        pc = self._class_prob(y, label=1)
        w0 = np.log(pc["pc1"]) - np.log(pc["pc2"])
        with np.errstate(divide="ignore"):
            w1 = np.log(qopt)
            idx_inf = np.isinf(w1)
            w1[idx_inf] = -33.6648265  # essentially the 0 entries, log(0) to -33.6648
            w2 = np.log(ropt)
            idx_inf = np.isinf(w2)
            w2[idx_inf] = -33.6648265
        w = w1 - w2

        # Return results
        return {"idx": idx, "w0": w0, "w": w, "q": qopt, "r": ropt, "objv": objv}


    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            raise ValueError(
                "k should be >=0, <= n_features = %d; got %r. "
                "Use k='all' to return all features." % (X.shape[1], self.k)
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
        k : target number of variables
        Returns
        -------
        self : object
        """
        # Check params
        X, y = check_X_y(X, y, ["csr", "csc"], multi_output=True)
        self._check_params(X, y)

        # Get features
        mask = np.zeros(X.shape[1], dtype=bool)
        if self.k == 0 or self.k == "all":
            scores = np.zeros(X.shape[1], dtype=bool)
        elif self._is_binary(X):
            res_nfs = self._binary_naive_feature_selection(X, y, self.k)
            mask[res_nfs["idx"]] = 1
            scores = np.square(res_nfs["w"])
        else:
            res_nfs = self._naive_feature_selection(X, y, self.k)
            mask[res_nfs["idx"]] = 1
            scores = np.square(res_nfs["w"])

        self.mask_ = mask
        self.scores_ = scores
        return self


    def _get_support_mask(self):
        check_is_fitted(self)
        return self.mask_


    def _is_binary(self,X):
        if issparse(X):
            return (X != X.astype(bool)).nnz==0
        if (np.array_equal(X, X.astype(bool))):
            return True
        return False

