import argschema
import alignment.solve_3d as s3
from alignment.transform import StagedTransform
import copy
import numpy as np
import os
import tempfile
from .schemas import StagedSolveSchema


class StagedSolve(argschema.ArgSchemaParser):
    default_schema = StagedSolveSchema

    def run(self):
        self.solves = []
        tmpfile = tempfile.NamedTemporaryFile()
        for i, transform in enumerate(self.args['transforms']):
            step_args = {}
            if i == 0:
                step_args['data'] = copy.deepcopy(self.args['data'])
                step_args['leave_out_index'] = self.args['leave_out_index']
            else:
                step_args['data'] = {
                        'landmark_file': tmpfile.name,
                        'header': [
                            'srcx', 'srcy', 'srcz',
                            'dstx', 'dsty', 'dstz'],
                        'sd_set': {'src': 'src', 'dst': 'dst'}
                        }

            step_args['output_json'] = os.path.join(
                    os.path.dirname(self.args['data']['landmark_file']),
                    "staged_transform_%d.json" % i)
            step_args['transform'] = transform
            self.solves.append(
                    s3.Solve3D(
                        input_data=copy.deepcopy(step_args),
                        args=[]))
            self.solves[-1].run()

            # write the transformed result to file
            # for input to the next stage
            s3.write_src_dst_to_file(
                    tmpfile.name,
                    self.solves[-1].transform.transform(
                        self.solves[-1].data['src']),
                    self.solves[-1].data['dst'])

        tmpfile.close()

        # this object combines the transforms
        # it converts input em units into the final units
        # through both transforms
        self.transform = StagedTransform([s.transform for s in self.solves])

        # let's convince ourselves it works
        total_tfsrc = self.transform.transform(self.solves[0].data['src'])
        self.residuals = self.solves[-1].data['dst'] - total_tfsrc
        # for 2p -> em this atol means the residuals are within 100nm
        # on any given axis, which is pretty good...
        diff = np.linalg.norm(
                self.residuals - self.solves[-1].residuals,
                axis=1)
        nhigh = np.count_nonzero(diff < 100)
        if nhigh != 0:
            self.logger.warning("{} of {} residuals exceed 100nm".format(
                nhigh, diff.size))

        # how far did the control points move for the thin plate part?
        for si, s in enumerate(self.solves):
            if s.transform.control_pts is not None:
                csrc = s.transform.control_pts
                print('control pts shape: ', csrc.shape)
                cdst = s.transform.transform(csrc)
                delta = cdst - csrc
                self.avdelta = np.linalg.norm(delta, axis=1).mean() * 0.001
                print('transform %d control points moved average of %0.1fum' %
                      (si, self.avdelta))

        self.leave_out_res = None
        if self.args['leave_out_index'] is not None:
            self.leave_out_res = (
                    self.solves[0].left_out['dst'] -
                    self.transform.transform(
                        self.solves[0].left_out['src'])).squeeze().tolist()
            self.leave_out_rmag = np.linalg.norm(self.leave_out_res)
            self.leave_out_label = self.solves[0].left_out['labels'][0]


if __name__ == '__main__':
    smod = StagedSolve()
    smod.run()
