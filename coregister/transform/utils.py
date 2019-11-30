import scipy
import numpy as np


def solve(A, w, r, dst):
    """regularized linear least squares

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        ndata x nparameter array
    w : :class:`numpy.ndarray`
        ndata x ndata diagonal weight matrix
    r : :class:`numpy.ndarray`
        nparameter x nparameter diagonal
        regularization matrix
    dst : :class:`numpy.ndarray`
        ndata x 3 Cartesian coordinates
        of transform destination.

    Returns
    -------
    x : :class:`numpy.ndarray`
        nparameter x 3 solution

    """
    ATW = A.transpose().dot(w)
    K = ATW.dot(A) + r
    lu, piv = scipy.linalg.lu_factor(K, overwrite_a=True)
    x = np.zeros((A.shape[1], dst.shape[1]))
    for i in range(dst.shape[1]):
        rhs = ATW.dot(dst[:, i])
        x[:, i] = scipy.linalg.lu_solve(
                (lu, piv), rhs)
    return x
