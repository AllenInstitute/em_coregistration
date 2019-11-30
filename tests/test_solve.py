from coregister.solve_3d import Solve3D
import numpy as np

example1 = {
        'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/transform.json',
        'data': {
            'landmark_file': './data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Master.csv',
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
            'landmark_file': './data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Master.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
        },
        "transform": {
            'name': 'PolynomialModel',
            'order': 1,
                }
        }


def test_example1(tmpdir):
    fname = str(tmpdir.mkdir("sub").join("transform.json"))
    s = Solve3D(input_data=example1, args=['--output_json', fname])
    s.run()
    rmag = np.linalg.norm(s.residuals, axis=1)
    assert rmag.mean() < 0.017

def test_example2(tmpdir):
    fname = str(tmpdir.mkdir("sub").join("transform.json"))
    s = Solve3D(input_data=example2, args=['--output_json', fname])
    s.run()
    rmag = np.linalg.norm(s.residuals, axis=1)
    assert rmag.mean() < 16000
