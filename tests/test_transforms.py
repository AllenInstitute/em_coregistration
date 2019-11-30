from coregister.transform import Transform, PolynomialModel
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


@pytest.mark.parametrize('order', [1, 2, 3])
def test_to_from_dict(order):
    t = Transform('PolynomialModel', order=order)
    t.parameters += np.random.randn(*t.parameters.shape) * 0.1
    jt = t.to_dict()

    t2 = Transform(json=jt)

    assert t2.order == t.order
    assert np.allclose(t2.parameters, t.parameters)

    tp = PolynomialModel(json=jt)
    assert tp.order == t.order
    assert np.allclose(tp.parameters, t.parameters)


@pytest.mark.parametrize('order', [0, 4])
def test_polynomial_order(order):
    with pytest.raises(ValueError):
        t = Transform(name="PolynomialModel", order=order)
