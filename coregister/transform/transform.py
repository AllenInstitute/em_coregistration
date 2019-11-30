import numpy as np
import scipy.spatial
from . affine import AffineModel

#def linear_kernel(src, control_pts=None):
#    """linear, i.e. affine, kernel
#
#    Parameters
#    ----------
#    src : :class:`numpy.ndarray`
#        npts x 3 Cartesian coordinates of
#        data points
#    control_pts : None
#        ignored for this kernel
#
#    Returns
#    -------
#    A : :class:`numpy.ndarray`
#        npts x 4 array
#
#    """
#    A = np.zeros((src.shape[0], 4))
#    A[:, 0] = 1.0  # offset
#    A[:, 1:4] = src
#    return A
#
#
#def polynomial_kernel(src, control_pts=None):
#    """2nd order polynomial kernel
#
#    Parameters
#    ----------
#    src : :class:`numpy.ndarray`
#        npts x 3 Cartesian coordinates of
#        data points
#    control_pts : None
#        ignored for this kernel
#
#    Returns
#    -------
#    A : :class:`numpy.ndarray`
#        npts x 10 array
#
#    """
#    A = np.zeros((src.shape[0], 10))
#    A[:, 0] = 1.0  # offset
#    A[:, 1:4] = src
#    A[:, 4] = src[:, 0]**2  # x^2
#    A[:, 5] = src[:, 1]**2  # y^2
#    A[:, 6] = src[:, 2]**2  # z^2
#    A[:, 7] = src[:, 0] * src[:, 1]  # xy
#    A[:, 8] = src[:, 0] * src[:, 2]  # xz
#    A[:, 9] = src[:, 1] * src[:, 2]  # yz
#    return A
#
#
#def thin_plate_kernel(src, control_pts=None):
#    """thin plate spline kernel, including affine
#
#    Parameters
#    ----------
#    src : :class:`numpy.ndarray`
#        npts x 3 Cartesian coordinates of
#        data points
#    control_pts : :class:`numpy.ndarray`
#        ncntrl x 3 Cartestion coordinates of
#        control points
#
#    Returns
#    -------
#    A : :class:`numpy.ndarray`
#        npts x (ncntrl + 4) array
#
#    """
#    A = np.zeros((src.shape[0], 4 + control_pts.shape[0]))
#    A[:, 0] = 1.0  # offset
#    A[:, 1:4] = src  # affine
#    # in 3 dimensions, the solution to the biharmonic
#    A[:, 4:] = np.abs(
#            scipy.spatial.distance.cdist(
#                src,
#                control_pts,
#                metric='euclidean'))
#
#    return A
#
#
#def chunked_kernel(src, nz=21, zr=None, axis='z'):
#    nax = ['x', 'y', 'z'].index(axis)
#
#    if zr is None:
#        zr = np.linspace(src[:, nax].min(), src[:, nax].max(), nz)
#        zr[0] -= zr.ptp() * 0.01
#        zr[-1] += zr.ptp() * 0.01
#
#    A = np.zeros((src.shape[0], 10 * (zr.size - 1)))
#    nrow = 0
#    for s in src:
#        for i in range(zr.size - 1):
#            if (s[nax] > zr[i]) & (s[nax] <= zr[i + 1]):
#                A[nrow, (i * 10):(i * 10 + 10)] = \
#                        polynomial_kernel(s.reshape(1, -1))
#                nrow += 1
#                break
#    return A, zr


class Transform():
    """class for creating solver matrix kernel and
       performing transformations. includes serialization
       and desrialization functions.
    """

    def __init__(self, name=None, json=None):
        """Initialize Transform
        Parameters
        ----------
        name : str
            classname of this transform
        json : dict
            json compatible representation of this transform
            (supersedes className, dataString, and transformId if not None)
        """
        classes = {
                "AffineModel": AffineModel
                }

        if json is not None:
            self.__class__ = classes[json['name']]

        elif name is not None:
            self.__class__ = classes[name]

        self.__class__.__init__(self, json=json)


    #def kernel(self, src, zr=None, tri=None):
    #    """matrix kernel from this Transform object

    #    Parameters
    #    ----------
    #    src : :class:`numpy.ndarray`
    #        ndata x 3 Cartesian coordinates of source data

    #    Returns
    #    -------
    #    A : :class:`numpy.ndarray`
    #        A matrix for this Transform, given src

    #    """

    #    if self.model == 'LIN':
    #        return linear_kernel(src)
    #    if self.model == 'ZCHUNKED':
    #        A, self.zr = chunked_kernel(src, nz=self.nz, zr=zr, axis=self.axis)
    #        return A
    #    elif self.model == 'POLY':
    #        return polynomial_kernel(src)
    #    elif self.model == 'TPS':
    #        return thin_plate_kernel(src, self.control_pts)

    #def load_parameters(self, x):
    #    """set the parameters of this Transform so that
    #       the transform() method can be called.

    #    Parameters
    #    ----------
    #    x : :class:`numpy.ndarray`
    #        nparameter x 3 parameters resulting from a solve

    #    """
    #    self.parameters = x

    #def transform(self, src):
    #    """transform a set of source points

    #    Parameters
    #    ----------
    #    src : :class:`numpy.ndarray`
    #        ndata x 3 Cartesian coordinates to transform

    #    Returns
    #    -------
    #    dst : :class:`numpy.ndarry`
    #        ndata x 3 transformed Cartesian coordinates

    #    """

    #    if self.model == 'ZCHUNKED':
    #        k = self.kernel(src, zr=self.zr)
    #    elif self.model == 'PIECEWISE':
    #        k = self.kernel(src, tri=self.tri)
    #    else:
    #        k = self.kernel(src)
    #    dst = np.zeros_like(src)
    #    for i in range(src.shape[1]):
    #        dst[:, i] = k.dot(self.parameters[:, i])
    #    return dst

    #def to_dict(self):
    #    """serialize this transform

    #    Returns
    #    -------
    #    dict representation of this transform

    #    """
    #    return {
    #            'model': self.model,
    #            'parameters': self.parameters.tolist(),
    #            'control_pts': self.control_pts.tolist()
    #            }

    #def from_dict(self, j):
    #    """set this transform via deserialization

    #    Parameters
    #    ----------
    #    j : dict
    #        result of previous to_dict() call

    #    """
    #    self.model = j['model']
    #    self.parameters = np.array(j['parameters'])
    #    self.control_pts = np.array(j['control_pts'])


#class StagedTransform():
#    def __init__(self, tflist):
#        self.tflist = tflist
#
#    def transform(self, coords):
#        x = np.copy(coords)
#        for tf in self.tflist:
#            x = tf.transform(x)
#        return x
