from coregister.transform import (
        Transform, PolynomialModel, ChunkedModel)
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
    noise = np.random.randn(*shape)
    dst = t.tform(src) + noise

    tfit = Transform(name="PolynomialModel", order=order)
    tfit.estimate(src, dst)
    tdst = tfit.tform(src)

    residuals = tdst - dst
    rmag = np.linalg.norm(residuals, axis=1)
    assert np.all(rmag < np.linalg.norm(noise, axis=1).max() * 2)


@pytest.mark.parametrize('order', [1, 2, 3])
def test_poly_to_from_dict(order):
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
        Transform(name="PolynomialModel", order=order)


@pytest.mark.parametrize('order', [1, 2, 3])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_chunked_identity(order, axis):
    t = Transform('ChunkedModel', order=order, axis=axis, nchunks=10)
    src = np.random.rand(10000, 3)
    t.estimate(src, src)
    dst = t.tform(src)
    assert np.allclose(src, dst)

    identity = Transform("PolynomialModel", order=order)
    for tf in t.transforms:
        if order == 3:
            assert np.allclose(
                    identity.parameters, tf.parameters, rtol=0, atol=1e-3)
        else:
            assert np.allclose(identity.parameters, tf.parameters)


def test_chunked_ranges():
    order = 1
    axis = 2
    t = Transform('ChunkedModel', order=order, axis=axis, nchunks=10)
    src = np.random.randn(1000, 3)
    t.estimate(src, src)

    t2 = Transform('ChunkedModel', order=order, axis=axis, ranges=t.ranges)
    assert t.nchunks == t2.nchunks
    assert np.allclose(t.ranges, t2.ranges)

    t2 = Transform('ChunkedModel', order=order, axis=axis, nchunks=t.nchunks)
    t2.set_ranges(ranges=t.ranges)
    assert t.nchunks == t2.nchunks
    assert np.allclose(t.ranges, t2.ranges)


def random_affine_parameters():
    p = np.random.randn(3) * 10
    p = np.vstack((p, np.eye(3)))
    p[1:4, :] += np.random.randn(3, 3) * 0.01
    return p


def test_chunked_values():
    order = 1
    axis = 2
    nchunks = 10
    src = np.random.randn(1000, 3)
    t1 = PolynomialModel(order=order, parameters=random_affine_parameters())
    t2 = PolynomialModel(order=order, parameters=random_affine_parameters())

    ranges = np.linspace(
            src[:, axis].min(),
            src[:, axis].max(),
            nchunks)[1:-1]

    inds_for_t2 = [3, 6, 7]

    dst = np.zeros_like(src)
    which = np.searchsorted(ranges, src[:, axis])
    for i in range(nchunks):
        inds = which == i
        tf = t1
        if i in inds_for_t2:
            tf = t2
        dst[inds, :] = tf.tform(src[inds, :])

    t = Transform('ChunkedModel', order=order, axis=axis, ranges=ranges)
    t.estimate(src, dst)

    tsrc = t.tform(src)
    res = np.linalg.norm(tsrc - dst, axis=1)
    assert res.mean() < np.linalg.norm(dst, axis=1).mean() * 0.001

    tc = ChunkedModel(order=order, axis=axis, ranges=ranges)
    tc.estimate(src, dst)

    tsrc = tc.tform(src)
    res = np.linalg.norm(tsrc - dst, axis=1)
    assert res.mean() < np.linalg.norm(dst, axis=1).mean() * 0.001

    # to and from dict
    jt = t.to_dict()
    td = Transform(json=jt)
    tsrc = td.tform(src)
    res = np.linalg.norm(tsrc - dst, axis=1)
    assert res.mean() < np.linalg.norm(dst, axis=1).mean() * 0.001
