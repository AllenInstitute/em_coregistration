from coregister.data_loader import DataLoader, invert_y, px_to_mm
import numpy as np
import pytest
import tempfile


@pytest.fixture(scope="module")
def somedata():
    data = np.random.randn(100, 6)
    yield data


@pytest.fixture(scope="module")
def somedata_in_file(somedata):
    with tempfile.NamedTemporaryFile() as temp:
        np.savetxt(temp.name, somedata, delimiter=",")
        yield temp.name


@pytest.fixture(scope="module")
def somedata_in_file_flagged(somedata):
    n = somedata.shape[0]
    flags = [True] * n
    for i in [0, 5, 7, 43]:
        flags[i] = False
    with tempfile.NamedTemporaryFile() as temp:
        with open(temp.name, 'w') as f:
            for i in range(n):
                f.write(('%s' + ',%f' * 6 + '\n') % (flags[i], *somedata[i]))
        yield temp.name


def test_basic(somedata, somedata_in_file):
    args = {
            "landmark_file": somedata_in_file,
            'header': ['emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
            }
    dl = DataLoader(input_data=args, args=[])
    dl.run()
    assert np.allclose(somedata[:, 0:3], dl.data['dst'])
    assert np.allclose(somedata[:, 3:], dl.data['src'])


def test_invert(somedata, somedata_in_file):
    data = np.copy(somedata)
    args = {
            "landmark_file": somedata_in_file,
            'header': ['emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'sd_set': {'src': 'opt', 'dst': 'em'},
            'actions': ['invert_opty']
            }
    dl = DataLoader(input_data=args, args=[])
    dl.run()
    assert np.allclose(data[:, 0:3], dl.data['dst'])
    assert np.allclose(data[:, 3], dl.data['src'][:, 0])
    assert np.allclose(data[:, 5], dl.data['src'][:, 2])
    assert np.allclose(invert_y(data[:, 4]), dl.data['src'][:, 1])


def test_to_px(somedata, somedata_in_file):
    data = np.copy(somedata)
    args = {
            "landmark_file": somedata_in_file,
            'header': ['emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'sd_set': {'src': 'opt', 'dst': 'em'},
            'actions': ['opt_px_to_mm']
            }
    dl = DataLoader(input_data=args, args=[])
    dl.run()
    assert np.allclose(data[:, 0:3], dl.data['dst'])
    assert np.allclose(px_to_mm(data[:, 3:]), dl.data['src'])


@pytest.mark.parametrize('allflags', [True, False])
def test_flagged(somedata, somedata_in_file_flagged, allflags):
    args = {
            "landmark_file": somedata_in_file_flagged,
            'header': ['flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'sd_set': {'src': 'opt', 'dst': 'em'},
            'actions': ['opt_px_to_mm']
            }
    if allflags:
        args['all_flags'] = True
    dl = DataLoader(input_data=args, args=[])
    dl.run()

    if allflags:
        assert somedata.shape[0] == dl.data['src'].shape[0]
    else:
        assert somedata.shape[0] == dl.data['src'].shape[0] + 4
