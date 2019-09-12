import alignment.solve_3d as s3
import copy
import numpy as np
import matplotlib.pyplot as plt

example = {
        "reg": 1e8,
        "npts": 10,
        "leave_out_index": None
        }

class StagedTransform():
    def __init__(self, tflist):
        self.tflist = tflist
    def transform(self, coords):
       x = np.copy(coords) 
       for tf in self.tflist:
           x = tf.transform(x)
       return x

class StagedSolve():
    def __init__(self, args):
        self.reg = args['reg']
        self.leave_out_index = args['leave_out_index']
        self.run()

    def run(self):
        # solve just with polynomial
        args_poly = copy.deepcopy(s3.example2)
        args_poly['model'] = 'POLY'
        args_poly['leave_out_index'] = self.leave_out_index
        s_poly = s3.Solve3D(input_data=args_poly, args=[])
        s_poly.run()
        tf_poly = s_poly.transform
        self.poly_residuals = s_poly.residuals
        # write the transformed result to file
        # for input to the next stage
        tmp_path = './tmp.csv'
        s3.write_src_dst_to_file(
                tmp_path,
                tf_poly.transform(s_poly.data['src']),
                s_poly.data['dst'])
        
        # solve with thin plate spline on top
        args_tps = copy.deepcopy(s3.example2)
        args_tps['model'] = 'TPS'
        args_tps['npts'] = 10
        args_tps['data'] = {
                'landmark_file': tmp_path,
                'header': ['polyx', 'polyy', 'polyz', 'emx', 'emy', 'emz'],
                'sd_set': {'src': 'poly', 'dst': 'em'}
                }
        args_tps['regularization']['other'] = self.reg
        s_tps = s3.Solve3D(input_data=args_tps, args=[])
        s_tps.run()
        tf_tps = s_tps.transform
        
        # this object combines the 2 transforms
        # it converts input em units into the final units
        # through both transforms
        self.transform = StagedTransform([tf_poly, tf_tps])

        # let's convince ourselves it works
        total_tfsrc = self.transform.transform(s_poly.data['src'])
        self.residuals = s_poly.data['dst'] - total_tfsrc
        # for 2p -> em this atol means the residuals are within 100nm
        # on any given axis, which is pretty good...
        assert np.all(np.isclose(self.residuals, s_tps.residuals, atol=100)) 

        # how far did the control points move for the thin plate part?
        csrc = tf_tps.control_pts
        cdst = tf_tps.transform(csrc)
        delta = cdst - csrc
        self.avdelta = np.linalg.norm(delta, axis=1).mean() * 0.001
        print('control points moved average of %0.1fum' % (self.avdelta))

        self.leave_out_res = None
        if self.leave_out_index is not None:
            self.leave_out_res = np.linalg.norm(
                    s_poly.left_out['dst'] - self.transform.transform(s_poly.left_out['src']))
            self.leave_out_label = s_poly.left_out['labels'][0]

def leave_one_out(reg, nmax, n):
    ind = np.arange(nmax)
    np.random.shuffle(ind)
    ind = ind[0:n]
    loo = {}
    for i in ind:
        args = dict(example)
        args['reg'] = r
        args['leave_out_index'] = i
        s = StagedSolve(args)
        loo[s.leave_out_label] = np.linalg.norm(s.leave_out_res)
    return loo

        
#reg = [1e18, 1e16, 1e14, 1e12, 1e11, 1e10, 1e9, 1e8, 1e7, 1e6]
reg = [1e10]
solves = []
leave_outs = []
leave_out_res = []
for r in reg:
    args = dict(example)
    args['reg'] = r
    solves.append(StagedSolve(args))
    leave_outs.append(leave_one_out(r, 2049, 2049))
leave_out_res = [np.array(list(l.values())).mean() * 0.001 for l in leave_outs]
residuals = [np.linalg.norm(s.residuals * 0.001, axis=1).mean() for s in solves]

tps_displacements = [s.avdelta for s in solves]
fig = plt.figure(1)
fig.clf()
ax1 = fig.add_subplot(111)
pres = np.linalg.norm(solves[0].poly_residuals, axis=1).mean() * 0.001
ax1.axhline(pres, linestyle=':', color='b', label='polynomial')
ax1.plot(reg, residuals, 'b', label='polynomial + spline')
ax1.plot(reg, leave_out_res,'--b', label='leave-one-out\npolynomial + spline')
ax1.tick_params('y', colors='b')
ax1.set_ylabel('Residuals [um]', fontsize=14, color='b')
ax1.set_xlabel('Spline Regularization', fontsize=14)
ax1.set_xscale('log')
ax1.legend(loc=5, title='mean residuals (left axis)')

ax2 = ax1.twinx()
ax2.plot(reg, tps_displacements, 'r')
ax2.tick_params('y', colors='r')
ax2.set_ylabel('Spline displacements [um]', fontsize=14, color='r')
