import argschema
from .schemas import SolverSchema
from .data_handler import DataLoader
from .transform.transform import Transform
import numpy as np
import scipy

example1 = {
        'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/transform.json',
        'data': {
            'landmark_file': './data/17797_2Pfix_EMmoving_20190910_1805.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty'],
            'sd_set': {'src': 'em', 'dst': 'opt'}
        },
        "transform": {
            'name': 'PolynomialModel',
            'order': 1
            }
        }

example2 = {
        'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/transform.json',
        'data': {
            'landmark_file': './data/17797_2Pfix_EMmoving_20190910_1805.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty', 'em_nm_to_neurog'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
        },
        "transform": {
            'name': 'PolynomialModel',
            'order': 1,
                }
        }


def control_pts_from_bounds(data, npts, bounds_buffer=0):
    """create thin plate spline control points
    from the bounds of provided data.

    Parameters
    ----------
    data : :class:`numpy.ndarray`
        ndata x 3 Cartesian coordinates of data.
    npts : list
        [nx, ny, nz]
        number of control points per axis. total
        number of control points will be nx * ny * nz

    Returns
    -------
    control_pts : :class:`numpy.ndarray`
        npts^3 x 3 Cartesian coordinates of controls.

    """
    x, y, z = [
            np.linspace(
                data[:, i].min() - bounds_buffer,
                data[:, i].max() + bounds_buffer,
                npts[i])
            for i in [0, 1, 2]]
    xt, yt, zt = np.meshgrid(x, y, z)
    control_pts = np.vstack((
        xt.flatten(),
        yt.flatten(),
        zt.flatten())).transpose()
    return control_pts


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
    solution = []
    x = np.zeros((A.shape[1], dst.shape[1]))
    for i in range(dst.shape[1]):
        rhs = ATW.dot(dst[:, i])
        x[:, i] = scipy.linalg.lu_solve(
                (lu, piv), rhs)
    return x


def create_regularization(n, d):
    """create diagonal regularization matrix

    Parameters
    ----------
    n : int
        number of parameters per Cartesian axis
    d : dict
        regularization dict from input

    Returns
    -------
    R : :class:`numpy.ndarray`
        n x n diagonal matrix containing regularization
        factors
        
    """
    r = np.ones(n)
    r[0] = d['translation']
    r[1:3] = d['linear']
    r[4:] = d['other']
    i = np.diag_indices(n)
    R = np.eye(n)
    R[i] = r
    return R


def write_src_dst_to_file(fpath, src, dst):
    """csv output of src and dst

    Parameters
    ----------
    fpath : str
        valid path
    src : :class:`numpy.ndarray`
        ndata x 3 source points
    dst : :class:`numpy.ndarray`
        ndata x 3 destination points

    """
    out = np.hstack((src, dst))
    np.savetxt(fpath, out, fmt='%0.8e', delimiter=',')


def list_points_by_res_mag(res, labels, n=np.inf, factor=0.001):
    """print to stdout point labels and residuals

    Parameters
    ----------
    res : :class:`numpy.ndarray`
        ndata x 3 residuals
    labels : list
        ndata length list of point labels
    n : int
        limit to print only highest n residuals
    factor : float
        scales the residuals

    """
    mag = np.linalg.norm(res, axis=1)
    ind = np.argsort(mag)[::-1]
    i = 0
    print('worst points:')
    while (i < n) & (i < ind.size):
        print('%10s, %0.1f' % (labels[ind][i], mag[ind][i] * factor))
        i += 1

def leave_out(data, index):
    if index is None:
        return data, None
    else:
        keep = np.ones(data['labels'].size).astype(bool)
        keep[index] = False
        kdata = {
                'src': data['src'][keep],
                'dst': data['dst'][keep],
                'labels': data['labels'][keep]
                }
        keep = np.invert(keep)
        ldata = {
                'src': data['src'][keep],
                'dst': data['dst'][keep],
                'labels': data['labels'][keep]
                }
        return kdata, ldata


class Solve3D(argschema.ArgSchemaParser):
    """class to solve a 3D coregistration problem"""
    default_schema = SolverSchema

    def run(self, control_pts=None):
        """run the solve

        Parameters
        ----------
        control_pts : :class:`numpy.ndarray`
            user-supplied ncntrl x 3 Cartesian coordinates
            of control points. default None will create
            control points from bounds of input data.

        """
        d = DataLoader(input_data=self.args['data'], args=[])
        d.run()
        self.data = d.data

        self.data, self.left_out = leave_out(self.data, self.args['leave_out_index'])

        self.transform = Transform(json=self.args['transform'])

        self.transform.estimate(self.data['src'], self.data['dst'])

        self.residuals = (
                self.data['dst'] -
                self.transform.tform(self.data['src']))

        print('average residual [dst units]: %0.4f' % (
            np.linalg.norm(self.residuals, axis=1).mean()))

        self.output(self.transform.to_dict(), indent=2)


if __name__ == '__main__':
    smod = Solve3D(input_data=example1)
    smod.run()
    smod = Solve3D(input_data=example2)
    smod.run()
