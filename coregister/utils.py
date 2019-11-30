import numpy as np

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
        vxyz = np.zeros_like(xyz).astype(float)
        vxyz[:, 0] = ((xyz[:, 0] / 4) - 3072)
        vxyz[:, 1] = ((xyz[:, 1] / 4) - 2560)
        vxyz[:, 2] = ((xyz[:, 2]/40.0) + 7924)
    return vxyz


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
