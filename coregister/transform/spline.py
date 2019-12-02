import numpy as np
import scipy.spatial
from . utils import solve


class SplineModel():
    def __init__(
            self, json=None, parameters=None,
            regularization=None, control_pts=None, ncntrl=[2, 2, 2]):

        if json is not None:
            self.from_dict(json)
            return

        self.ncntrl = np.array(ncntrl)
        if control_pts is not None:
            self.control_pts = np.array(control_pts)

        if parameters is None:
            self.set_identity_parameters()
        elif type(parameters) == list:
            self.parameters = np.array(parameters)

        self.set_regularization(regularization)

    def set_identity_parameters(self):
        self.parameters = np.array([
            [0.0, 0.0, 0.0]])
        self.parameters = np.vstack((
            self.parameters,
            np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0]])))
        self.parameters = np.vstack((
            self.parameters,
            np.zeros((self.ncntrl.prod(), 3))))

    def set_control_pts_from_src(self, src, ncntrl=None):
        if ncntrl is not None:
            self.ncntrl = np.array(ncntrl)
        x, y, z = [
                np.linspace(
                    src[:, i].min(), src[:, i].max(), self.ncntrl[i] + 2)[1:-1]
                for i in [0, 1, 2]]
        xt, yt, zt = np.meshgrid(x, y, z)
        self.control_pts = np.vstack((
            xt.flatten(),
            yt.flatten(),
            zt.flatten())).transpose()

    def set_regularization(self, regularization=None):
        if regularization is None:
            regularization = 0.0
        if type(regularization) in [list, np.ndarray]:
            self.regularization = np.array(regularization)
        else:
            self.regularization = np.array(
                    [regularization] * self.parameters.shape[0])

    def from_dict(self, json):
        self.ncntrl = np.array(json['ncntrl'])
        if 'control_pts' in json:
            self.control_pts = np.array(json['control_pts'])
        if 'parameters' in json:
            self.parameters = np.array(json['parameters'])
        else:
            self.set_identity_parameters()
        regularization = None
        if 'regularization' in json:
            regularization = json['regularization']
            self.set_regularization(regularization=regularization)

    def to_dict(self):
        return {
                'name': "SplineModel",
                'ncntrl': self.ncntrl.tolist(),
                'control_pts': self.control_pts.tolist(),
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
        A = np.zeros((src.shape[0], 4 + self.control_pts.shape[0]))
        A[:, 0] = 1.0  # offset
        A[:, 1:4] = src  # affine
        # in 3 dimensions, the solution to the biharmonic
        A[:, 4:] = np.abs(
                scipy.spatial.distance.cdist(
                    src,
                    self.control_pts,
                    metric='euclidean'))
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
        if not hasattr(self, 'control_pts'):
            self.set_control_pts_from_src(src, self.ncntrl)
        self.parameters = solve(
                self.kernel(src),
                wts,
                self.regularization,
                dst)
