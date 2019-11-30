import numpy as np
import itertools
from . utils import solve


class PolynomialModel():
    def __init__(self, json=None, parameters=None, regularization=0, order=1):
        ndict = {
                1: 4,
                2: 10,
                3: 20
                }
        self.order = order
        if order not in ndict:
            raise ValueError(
                    "order = {} not currently supported".format(order))

        if type(regularization) in [list, np.ndarray]:
            self.regularization = np.array(regularization)
        else:
            self.regularization = np.array(
                    [regularization] * ndict[self.order])

        if parameters is None:
            self.parameters = np.array([
                [0.0, 0.0, 0.0]])
            if self.order > 0:
                self.parameters = np.vstack((
                    self.parameters,
                    np.array([
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0]])))
            if self.order > 1:
                self.parameters = np.vstack((
                    self.parameters,
                    np.zeros((ndict[self.order] - 4, 3))))

        else:
            self.parameters = np.array(parameters)

        if json is not None:
            self.from_dict(json)

    def from_dict(self, json):
        self.order = json['order']
        if 'parameters' in json:
            self.parameters = np.array(json['parameters'])
        if 'regularization' in json:
            self.regularization = np.array(json['regularization'])

    def to_dict(self):
        return {
                'name': "PolynomialModel",
                'order': self.order,
                'parameters': self.parameters.tolist(),
                'regularization': self.regularization.tolist()
                }

    def kernel(self, src):
        """linear, i.e. affine, kernel

        Parameters
        ----------
        src : :class:`numpy.ndarray`
            npts x 3 Cartesian coordinates of
            data points

        Returns
        -------
        A : :class:`numpy.ndarray`
            npts x 4 array

        """
        ones = np.ones(src.shape[0]).reshape(-1, 1)
        A = np.copy(ones)
        for order in range(1, self.order + 1):
            combos = itertools.combinations_with_replacement(range(3), order)
            for combo in combos:
                A = np.hstack((A, np.copy(ones)))
                for ind in combo:
                    A[:, -1] *= src[:, ind]
        return A

    def tform(self, src):
        """transform a set of source points

        Parameters
        ----------
        src : :class:`numpy.ndarray`
            ndata x 3 Cartesian coordinates to transform

        Returns
        -------
        dst : :class:`numpy.ndarry`
            ndata x 3 transformed Cartesian coordinates

        """
        k = self.kernel(src)
        dst = np.vstack([k.dot(p) for p in self.parameters.T]).T
        return dst

    def estimate(self, src, dst, wts=None):
        if wts is None:
            wts = np.eye(src.shape[0])
        self.parameters = solve(
                self.kernel(src),
                wts,
                self.regularization,
                dst)
