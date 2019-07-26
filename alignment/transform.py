import numpy as np
import scipy.spatial


def em_nm_to_voxels(xyz, inverse=False):
    """convert EM nanometers to neuroglancer voxels

    Parameters
    ----------
    xyz : :class:`numpy.ndarray`
        N x 3, the inut array in nm
    inverse : bool
        go from voxels to nm

    Returns
    -------
    vxyz : :class:`numpy.ndarray`
        N x 3, the output array in voxels

    """
    if inverse:
        vxyz = np.zeros_like(xyz).astype(float)
        vxyz[:, 0] = (xyz[:, 0] + 3072) * 4.0
        vxyz[:, 1] = (xyz[:, 1] + 2560) * 4.0
        vxyz[:, 2] = (xyz[:, 2] - 7924) * 40.0
    else:
        vxyz = np.zeros_like(xyz).astype(int)
        vxyz[:, 0] = ((xyz[:, 0] / 4) - 3072).astype('int')
        vxyz[:, 1] = ((xyz[:, 1] / 4) - 2560).astype('int')
        vxyz[:, 2] = ((xyz[:, 2]/40.0) + 7924).astype('int')
    return vxyz


def linear_kernel(src, control_pts=None):
    """linear, i.e. affine, kernel

    Parameters
    ----------
    src : :class:`numpy.ndarray`
        npts x 3 Cartesian coordinates of
        data points
    control_pts : None
        ignored for this kernel

    Returns
    -------
    A : :class:`numpy.ndarray`
        npts x 4 array 

    """
    A = np.zeros((src.shape[0], 4))
    A[:, 0] = 1.0 # offset
    A[:, 1:4] = src
    return A


def polynomial_kernel(src, control_pts=None):
    """2nd order polynomial kernel

    Parameters
    ----------
    src : :class:`numpy.ndarray`
        npts x 3 Cartesian coordinates of
        data points
    control_pts : None
        ignored for this kernel

    Returns
    -------
    A : :class:`numpy.ndarray`
        npts x 10 array 

    """
    A = np.zeros((src.shape[0], 10))
    A[:, 0] = 1.0 # offset
    A[:, 1:4] = src
    A[:, 4] = src[:, 0]**2 # x^2
    A[:, 5] = src[:, 1]**2 # y^2
    A[:, 6] = src[:, 2]**2 # z^2
    A[:, 7] = src[:, 0] * src[:, 1] # xy
    A[:, 8] = src[:, 0] * src[:, 2] # xz
    A[:, 9] = src[:, 1] * src[:, 2] # yz
    return A


def thin_plate_kernel(src, control_pts=None, old=False):
    """thin plate spline kernel, including affine

    Parameters
    ----------
    src : :class:`numpy.ndarray`
        npts x 3 Cartesian coordinates of
        data points
    control_pts : :class:`numpy.ndarray`
        ncntrl x 3 Cartestion coordinates of
        control points

    Returns
    -------
    A : :class:`numpy.ndarray`
        npts x (ncntrl + 4) array

    """
    A = np.zeros((src.shape[0], 4 + control_pts.shape[0]))
    A[:, 0] = 1.0 # offset
    A[:, 1:4] = src # affine
    if old:
        # this was a mistake, using the 2D solution to the biharmonic
        disp = scipy.spatial.distance.cdist(
            src,
            control_pts,
            metric='sqeuclidean')
        disp *= np.ma.log(np.sqrt(disp)).filled(0.0)
        A[:, 4:] = disp
    else:
        # in 3 dimensions, the solution to the biharmonic
        A[:, 4:] = np.abs(
                scipy.spatial.distance.cdist(
                    src,
                    control_pts,
                    metric='euclidean'))
    return A


class Transform():
    """class for creating solver matrix kernel and
       performing transformations. includes serialization
       and desrialization functions.
    """

    def __init__(self, model, control_pts=None):
        self.control_pts = control_pts
        self.model = model

    def kernel(self, src):
        """matrix kernel from this Transform object

        Parameters
        ----------
        src : :class:`numpy.ndarray`
            ndata x 3 Cartesian coordinates of source data

        Returns
        -------
        A : :class:`numpy.ndarray`
            A matrix for this Transform, given src

        """

        if self.model == 'LIN':
            return linear_kernel(src)
        elif self.model == 'POLY':
            return polynomial_kernel(src)
        elif self.model == 'TPS':
            return thin_plate_kernel(src, self.control_pts)
        elif self.model == 'TPS-wrong':
            return thin_plate_kernel(src, self.control_pts, old=True)

    def load_parameters(self, x):
        """set the parameters of this Transform so that
           the transform() method can be called.

        Parameters
        ----------
        x : :class:`numpy.ndarray`
            nparameter x 3 parameters resulting from a solve

        """
        self.parameters = x

    def transform(self, src):
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
        dst = np.zeros_like(src)
        for i in range(src.shape[1]):
            dst[:, i] = k.dot(self.parameters[:, i])
        return dst

    def to_dict(self):
        """serialize this transform

        Returns
        -------
        dict representation of this transform

        """
        return {
                'model': self.model,
                'parameters': self.parameters.tolist(),
                'control_pts': self.control_pts.tolist()
                }

    def from_dict(self, j):
        """set this transform via deserialization

        Parameters
        ----------
        j : dict
            result of previous to_dict() call

        """
        self.model = j['model']
        self.parameters = np.array(j['parameters'])
        self.control_pts = np.array(j['control_pts'])
