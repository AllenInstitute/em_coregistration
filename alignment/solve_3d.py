import argschema
from .schemas import SolverSchema
from .data_handler import DataLoader
from .transform import Transform
import numpy as np
import scipy

example1 = {
        'data': {
            'landmark_file': './data/17797_2Pfix_EMmoving_20190405_PA_1724_merged.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty'],
            'sd_set': {'src': 'em', 'dst': 'opt'}
        },
        'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/3D_alignment/tmp_out/transform.json',
        'model': 'TPS',
        'npts': 10,
        'regularization': {
            'translation': 1e0,
            'linear': 1e0,
            'other': 1e20
            }
}
example2 = {
        'data': {
            'landmark_file': './data/17797_2Pfix_EMmoving_20190405_PA_1724_merged.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
        },
        'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/3D_alignment/tmp_out/transform.json',
        'model': 'TPS',
        'npts': 10,
        'regularization': {
            'translation': 1e-10,
            'linear': 1e-3,
            'other': 1e-3,
            }
}


def control_pts_from_bounds(data, npts):
    x, y, z = [
            np.linspace(data[:, i].min(), data[:, i].max(), npts)
            for i in [0, 1, 2]]
    xt, yt, zt = np.meshgrid(x, y, z)
    control_pts = np.vstack((
        xt.flatten(),
        yt.flatten(),
        zt.flatten())).transpose()
    return control_pts


def solve(A, w, r, dst):
    ATW = A.transpose().dot(w)
    K = ATW.dot(A) + r
    lu, piv = scipy.linalg.lu_factor(K, overwrite_a=True)
    solution = []
    x = np.zeros((A.shape[1], dst.shape[1]))
    for i in range(dst.shape[1]):
        rhs = ATW.dot(dst[:, i])
        x[:, i] = scipy.linalg.lu_solve(
                (lu, piv), rhs)
    return x


def create_regularization(n, d):
    r = np.ones(n)
    r[0] = d['translation']
    r[1:3] = d['linear']
    r[4:] = d['other']
    i = np.diag_indices(n)
    R = np.eye(n)
    R[i] = r
    return R


def write_control_to_file(fpath, src, dst):
    out = np.hstack((src, dst))
    np.savetxt(fpath, out, fmt='%0.8e', delimiter=',')
    print('wrote %s' % fpath)


class Solve3D(argschema.ArgSchemaParser):
    default_schema = SolverSchema

    def run(self, control_pts=None):
        d = DataLoader(input_data=self.args['data'], args=[])
        d.run()
        self.data = d.data
        print(self.data.keys())

        if control_pts is None:
            if self.args['npts']:
                control_pts = control_pts_from_bounds(
                        self.data['src'],
                        self.args['npts'])

        self.transform = Transform(
                self.args['model'], control_pts=control_pts)

        # unit weighting per point
        self.wts = np.eye(self.data['src'].shape[0])

        self.A = self.transform.kernel(self.data['src'])

        self.reg = create_regularization(
                self.A.shape[1], self.args['regularization'])

        # solve the system of equations
        self.x = solve(
                self.A,
                self.wts,
                self.reg,
                self.data['dst'])

        # set the parameters for the transform
        self.transform.load_parameters(self.x)

        self.residuals = (
                self.data['dst'] -
                self.transform.transform(self.data['src']))

        print(np.linalg.norm(self.residuals, axis=1).mean())

        self.output(self.transform.to_dict(), indent=2)


if __name__ == '__main__':
    smod = Solve3D(input_data=example1)
    smod.run()
