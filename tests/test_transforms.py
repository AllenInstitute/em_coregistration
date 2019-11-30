from coregister.transform.transform import Transform
import numpy as np

def test_AffineModel():
    # identity
    t = Transform(name="AffineModel")
    src = np.random.randn(100, 3)
    assert np.allclose(t.tform(src), src)
    assert np.allclose(t.inverse_tform(src), src)

    # translations
    trans = np.random.randn(3) * 100
    t.parameters[3, 0:3] = trans
    dst = t.tform(src)
    assert np.allclose((src + trans), dst)
    isrc = t.inverse_tform(dst)
    assert np.allclose(isrc, src)

    # non-translations
    t = Transform(name="AffineModel")
    t.parameters[0:3, 0:3] = np.eye(3) + np.random.randn(3, 3) * 0.1
    dst = t.tform(src)
    isrc = t.inverse_tform(dst)

    # everything
    t = Transform(name="AffineModel")
    t.parameters[0:3, 0:3] = np.eye(3) + np.random.randn(3, 3) * 0.1
    t.parameters[3, 0:3] = np.random.randn(3) * 1000
    dst = t.tform(src)
    isrc = t.inverse_tform(dst)
    assert np.allclose(isrc, src)

    # json
    j = t.to_dict()
    t2 = Transform(json=j)
    dst = t2.tform(src)
    isrc = t2.inverse_tform(dst)
    assert np.allclose(isrc, src)
