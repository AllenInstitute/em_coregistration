from coregister.utils import em_nm_to_voxels, write_src_dst_to_file
import pytest
import numpy as np


@pytest.mark.parametrize("inverse", [True, False])
def test_em_nm_to_voxels(inverse):
    xyz = np.random.rand(1000, 3) * 1e6
    tf = em_nm_to_voxels(xyz, inverse=inverse)
    assert np.all(tf.shape == xyz.shape)
    itf = em_nm_to_voxels(tf, inverse=(not inverse))
    assert np.allclose(xyz.astype('int'), itf.astype('int'))


def test_write_src_dst(tmpdir):
    src = np.random.randn(100, 3)
    dst = np.random.randn(100, 3)
    fname = str(tmpdir.mkdir("sub").join("test.txt"))
    write_src_dst_to_file(fname, src, dst)
    data = np.loadtxt(fname, delimiter=",")
    assert np.allclose(src, data[:, 0:3])
    assert np.allclose(dst, data[:, 3:])
