import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import GridSearchCV


_log_0 = -33.6648265


def _split_classes(x, y, label=0, mask: bool = False):  # TODO: Use scikit learn here
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
    mask1 = y == label
    mask2 = y != label
    if mask:
        return mask1, mask2
    else:
        return x[mask1, :], x[mask2, :]


def _class_prob(y, label=0):  # TODO: Use scikit-learn here
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
    s = np.sum(y == label)
    return s / len(y), 1 - s / len(y)


def _largest_indices(ar, k):
    """
    Computes the k largest indices of a given array.
    :param ar: Array whose largest indices have to be computed
    :param k: Number of indices to compute
    :return: The k largest indices of ar
    """
    return np.argpartition(ar, -k)[-k:]


def negative_entropy(x):
    """
    Negative entropy function.

    This returns sum_i x_i * log(x_i) where log(x) = 0 if x < 0.
    """
    res = np.zeros(x.shape)
    ind = x > 0
    res[ind] = x[ind] * np.log(x[ind])
    return res


def auxiliary_gradient(alpha, k, c, f1, f2):
    """
    Auxiliary gradient function.
    """
    h = c - f1 * np.log(alpha) - f2 * np.log(1 - alpha)
    idx = _largest_indices(h, k)
    # fval = np.sum(h[idx])
    grad_h = -f1 / alpha + f2 / (1 - alpha)
    grad_phi = np.sum(grad_h[idx])
    return grad_phi, idx


def bisect_phi(f1, f2, k, tol: float, alpha_stop: float, f=None, c=None):
    """
    Dual objective function to be optimized. Bisection method is used.
    """
    if f is None:
        f = f1 + f2
    if c is None:
        c = negative_entropy(f1) + negative_entropy(f2) - negative_entropy(f)
    alpha_low, alpha_top = 0, 1
    while True:
        alpha = (alpha_top + alpha_low) / 2.0
        grad_phi, idx = auxiliary_gradient(alpha, k, c, f1, f2)
        if (alpha_top - alpha_low) <= alpha_stop:
            break
        if grad_phi > tol:
            alpha_top = alpha
        elif grad_phi < -tol:  # TODO: check if OK stopping on gradient?
            alpha_low = alpha
        else:
            break
    return idx


def reconstruct_primal_point(f1, f2, idx, f_sum=None):
    """
    Reconstruct a primal feasible (sub-optimal) point.
    :param f1: Naive data average (positive class)
    :param f2: Naive data average (negative class)
    :param idx: Set of indices of the top k entries of h(alpha*)
    :param f_sum: sum(f1 + f2) Can be passed if pre-computed
    :return:
    """
    if f_sum is None:
        f_sum = np.sum(f1, + f2)

    mask = np.ones(len(f1), dtype=bool)  # all elements included/True.
    mask[idx] = False

    q_opt = np.zeros(len(f1))
    r_opt = np.zeros(len(f2))

    q_opt[mask] = (f1[mask] + f2[mask]) / f_sum
    r_opt[mask] = q_opt[mask]

    q_opt[idx] = np.sum(f1[idx] + f2[idx]) * f1[idx] / (np.sum(f1[idx]) * f_sum)
    r_opt[idx] = np.sum(f1[idx] + f2[idx]) * f2[idx] / (np.sum(f2[idx]) * f_sum)

    return q_opt, r_opt


def find_linear_classification_rule(q_opt, r_opt, pc1, pc2):
    """
    Finds the linear classification rule w_0 + w' * w > 0
    :return:
    """
    w_0 = np.log(pc1) - np.log(pc2)
    with np.errstate(divide='ignore'):
        w1 = np.log(q_opt)
        idx_inf = np.isinf(w1)
        w1[idx_inf] = _log_0  # essentially the 0 entries, log(0) to -33.6648
        w2 = np.log(r_opt)
        idx_inf = np.isinf(w2)
        w2[idx_inf] = _log_0
    w = w1 - w2
    return w_0, w


def sparse_naive_bayes_from_averages(
        f1: np.ndarray, f2: np.ndarray, pc1: float, pc2: float, k: int,
        tol: float = 1e-6, alpha_stop: float = 1e-10, f=None, f_sum=None, c=None,
):
    if f is None:
        f = f1 + f2
    if f_sum is None:
        f_sum = np.sum(f)

    idx = bisect_phi(f1, f2, k, tol, alpha_stop, f=f, c=c)

    q_opt, r_opt = reconstruct_primal_point(f1, f2, idx, f_sum=f_sum)

    objv = f1[f1 != 0] @ np.log(q_opt[f1 != 0]) + f2[f2 != 0] @ np.log(r_opt[f2 != 0])

    w_0, w = find_linear_classification_rule(q_opt, r_opt, pc1, pc2)

    return {
        'idx': idx,
        'q': q_opt,
        'r': r_opt,
        'objv': objv,
        'w0': w_0,
        'w': w
    }


def naive_bayes_data_averages(x, y, alpha: float = 1e-10):
    """
    Computes the required data averages to run sparse naive bayes.
    :param x: Feature matrix
    :param y: Label vector
    :param alpha: Laplace smoothing parameter
    :return: (f1, f2, pc1, pc2) where
        - f1 is the sum of the lines of x with positive label
        - f2 is the sum of the lines of x with negative label
        - pc1 is an estimation of the probability of the positive class
        - pc2 is an estimation of the probability of the negative class
    """
    split1, split2 = _split_classes(x, y, label=1)
    c1, c2 = split1.shape[0], split2.shape[0]

    f1 = np.squeeze(np.asarray(np.sum(split1, axis=0))) + alpha * c1
    f2 = np.squeeze(np.asarray(np.sum(split2, axis=0))) + alpha * c2

    pc1, pc2 = _class_prob(y, label=1)

    return f1, f2, pc1, pc2


def fast_naive_bayes_data_averages(x, y, alpha: float = 1e-10):
    mask1, mask2 = _split_classes(x, y, label=1, mask=True)

    pc1 = np.mean(mask1)
    pc2 = 1 - pc1

    # The argument 'where' works on the last axis. Transposing the array that way
    # allows to sum the right indices without copying x in memory.
    f1 = np.sum(x.T, axis=1, where=mask1).T + alpha * pc1 * x.shape[0]
    f2 = np.sum(x.T, axis=1, where=mask2).T + alpha * pc2 * x.shape[0]

    return f1, f2, pc1, pc2


def sparse_naive_bayes(x: np.ndarray, y: np.ndarray, k: int, alpha: float = 1e-10, **kwargs):
    """
    Sparse Naive Bayes

    Description: Function that solves the following multinomial naive Bayes
    training problem, written

    max_{q,r}   fp^T log q + fm^T log r

        s.t. 	||q - r||_0 <= k
                0 <= q, r
                sum(q) = 1, sum(r) = 1

    Arguments:
        - X: training data as a n x m matrix (n data points with m features per data point)
        - Y: binary class labels for the n different points (one of the labels should be 1)
        - k: target cardinality for vector in linear classification rule
        - alpha: Laplacian smoothing parameter

    Returns:
        - idx: Index of selected features
        - q, r: Solutions reconstructed from optimal dual solution
        - w0, w: Linear classification rule: w0 + w' * x > 0, where w is k sparse.
    """

    f1, f2, pc1, pc2, = naive_bayes_data_averages(x, y, alpha=alpha)

    return sparse_naive_bayes_from_averages(f1, f2, pc1, pc2, k, **kwargs)


def f_stats(f1, f2):
    f = f1 + f2
    return (
            f,
            np.sum(f),
            negative_entropy(f1) + negative_entropy(f2) - negative_entropy(f)
    )


def sparse_naive_bayes_path(x, y, alpha: float = 1e-10):
    f1, f2, pc1, pc2 = naive_bayes_data_averages(x, y, alpha=alpha)
    f, f_sum, c = f_stats(f1, f2)
    ks = np.arange(1, x.shape[1] + 1)
    ws = []
    w0s = []
    for k in ks:
        snb = sparse_naive_bayes_from_averages(f1, f2, pc1, pc2, k, f=f, f_sum=f_sum, c=c)
        w0s.append(snb['w0'])
        ws.append(snb['w'])
    return np.array(ws)


class SparseMultinomialNB(BaseEstimator, ClassifierMixin):

    def __init__(self, k: int = 1, alpha: float = 1e-10, remove_min: bool = True):
        self.k = k
        self.alpha = alpha
        self.remove_min = remove_min

    def fit(self, x, y):
        if self.remove_min:
            x = x - np.min(x)
        snb = sparse_naive_bayes(x, y, k=self.k, alpha=self.alpha)
        self.intercept_, self.coef_ = snb['w0'], snb['w']

        return self

    def predict(self, x):
        check_is_fitted(self, ['intercept_', 'coef_'])
        return self.intercept_ + x @ self.coef_ >= 0


class SparseMultinomialNBCV:

    def __init__(
            self, delta: int = 1, alpha: float = 1e-10, cv: int = 3, remove_min: bool = True,
            alphas: np.ndarray = None
    ):
        self.delta = delta
        self.alpha = alpha
        self.alphas = alphas
        self.cv = cv
        self.remove_min = remove_min

    def fit(self, x, y):
        if self.remove_min:
            x = x - np.min(x)

        param_grid = {
            'k': np.arange(1, x.shape[1], self.delta)
        }
        if self.alphas is not None:
            param_grid['alpha'] = self.alphas

        grid_search = GridSearchCV(
            SparseMultinomialNB(alpha=self.alpha),
            param_grid=param_grid,
            cv=self.cv,
            iid=False
        )
        grid_search.fit(x, y)
        self.coef_ = grid_search.best_estimator_.coef_
        self.alpha_ = grid_search.best_estimator_.alpha
        self.k_ = grid_search.best_estimator_.k

        return self
