import numpy as np


class AffineModel():
    def __init__(self, json=None):
        if json is not None:
            self.from_dict(json)
        else:
            self.parameters = np.array([
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0]])

    def from_dict(self, json):
        self.parameters = np.array(json['parameters'])

    def to_dict(self):
        return {
                'name': "AffineModel",
                'parameters': self.parameters.tolist()
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
        A = np.hstack((src, np.ones(src.shape[0]).reshape(-1, 1)))
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

    def inverse_tform(self, src):
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
        src_inv_trans = src - self.parameters[3, :]
        ipar = np.linalg.inv(self.parameters[0:3, 0:3].T)
        dst = ipar.dot(src_inv_trans.T).T
        return dst
