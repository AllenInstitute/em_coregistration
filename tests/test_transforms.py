from coregister.transform import em_nm_to_voxels
import pytest
import numpy as np

@pytest.mark.parametrize("inverse", [True, False])
def test_em_nm_to_voxels(inverse):
   xyz = np.random.rand(1000, 3) * 1e6
   tf = em_nm_to_voxels(xyz, inverse=inverse)
   assert np.all(tf.shape == xyz.shape)
   itf = em_nm_to_voxels(tf, inverse=(not inverse))
   assert np.allclose(xyz.astype('int'), itf.astype('int'))
