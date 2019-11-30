from coregister.transform.transform import Transform
import numpy as np
import pytest


@pytest.mark.parametrize('order', [1, 2, 3])
def test_polynomial_identity(order):
    t = Transform(name="PolynomialModel", order=order)
    src = np.random.randn(100, 3)
    assert np.allclose(t.tform(src), src)


@pytest.mark.parametrize('order', [1, 2, 3])
def test_polynomial_solution(order):
    shape = (100, 3)
    src = np.random.randn(*shape) * 1000
    t = Transform(name="PolynomialModel", order=1)
    t.parameters = np.array([
        [100, -234, 456],
        [1.01, 0.01, -0.01],
        [-0.002, 0.97, 0.05],
        [-0.03, 0.001, 1.02]])
    dst = t.tform(src) + np.random.randn(*shape)

    tfit = Transform(name="PolynomialModel", order=order)
    tfit.estimate(src, dst)
    tdst = tfit.tform(src)

    residuals = tdst - dst
    rmag = np.linalg.norm(residuals, axis=1)
    assert np.all(rmag < 5)


@pytest.mark.parametrize('order', [0, 4])
def test_polynomial_order(order):
    with pytest.raises(ValueError):
        t = Transform(name="PolynomialModel", order=order)



    ## translations
    #trans = np.random.randn(3) * 100
    #t.parameters[0, 0:3] = trans
    #dst = t.tform(src)
    #assert np.allclose((src + trans), dst)
    #isrc = t.inverse_tform(dst)
    #assert np.allclose(isrc, src)

    ## non-translations
    #t = Transform(name="AffineModel")
    #t.parameters[1:4, :] = np.eye(3) + np.random.randn(3, 3) * 0.1
    #dst = t.tform(src)
    #isrc = t.inverse_tform(dst)

    ## everything
    #t = Transform(name="AffineModel")
    #t.parameters[1:4, :] = np.eye(3) + np.random.randn(3, 3) * 0.1
    #t.parameters[0, 0:3] = np.random.randn(3) * 1000
    #dst = t.tform(src)
    #isrc = t.inverse_tform(dst)
    #assert np.allclose(isrc, src)

    ## json
    #j = t.to_dict()
    #t2 = Transform(json=j)
    #dst = t2.tform(src)
    #isrc = t2.inverse_tform(dst)
    #assert np.allclose(isrc, src)
