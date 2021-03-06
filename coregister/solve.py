import argschema
from .schemas import SolverSchema
from .data_loader import DataLoader
from .transform.transform import Transform
import numpy as np
import copy


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
            'actions': ['invert_opty', 'em_nm_to_neurog'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
        },
        "transform": {
            'name': 'PolynomialModel',
            'order': 1,
                }
        }


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

        self.data, self.left_out = leave_out(
                self.data, self.args['leave_out_index'])

        self.transform = Transform(json=self.args['transform'])

        self.transform.estimate(self.data['src'], self.data['dst'])

        self.residuals = (
                self.data['dst'] -
                self.transform.tform(self.data['src']))
        self.residual_mag = np.linalg.norm(self.residuals, axis=1)

        if self.left_out is not None:
            self.leave_out_res = (
                    self.left_out['dst'] - 
                    self.transform.tform(self.left_out['src']))
            self.leave_out_rmag = np.linalg.norm(self.leave_out_res, axis=1)

        print('average residual [dst units]: %0.4f' % (
            self.residual_mag.mean()))

        inds = np.argsort(self.residual_mag)
        self.sorted_labeled_residuals = [(self.data['labels'][i], self.residual_mag[i]) for i in inds]

        self.output(self.transform.to_dict(), indent=2)

    def predict_all_data(self):
        alldata = DataLoader(
                input_data=copy.deepcopy(self.args['data']),
                args=['--all_flags', 'True'])
        alldata.run()
        inds = alldata.data['flag'] == False
        alldata.data['dst'][inds] = self.transform.tform(
                alldata.data['src'][inds])
        return alldata


if __name__ == '__main__':  # pragma: no cover
    smod = Solve3D(input_data=example1)
    smod.run()
