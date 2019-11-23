from alignment.staged_solve import StagedSolve
from alignment.data_handler import DataLoader, invert_y
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import itertools
import multiprocessing
import json


with open("./data/staged_transform_args.json", "r") as f:
    args = json.load(f)


s = StagedSolve(input_data=copy.deepcopy(args), args=[])
s.run()

rlist = s.sorted_labeled_residuals()
for r in rlist[0:10]:
    print("%10s %10.1f" % (r[0], r[1]))

alldata = s.predict_all_data()
alldata.data['src'][:, 1] = invert_y(alldata.data['src'][:, 1])

# write an updated file
new_name = os.path.join(
        os.path.dirname(s.args['data']['landmark_file']),
        os.path.splitext(
            os.path.basename(s.args['data']['landmark_file']))[0] +
        '_updated.csv')
fmt = '"%s","%s","%0.1f","%0.1f","%0.1f","%0.6f","%0.6f","%0.6f"\n'
fstring = ""
for i in range(alldata.data['src'].shape[0]):
    fstring += fmt % (
       alldata.data['labels'][i],
       alldata.data['flag'][i],
       *alldata.data['dst'][i],
       *alldata.data['src'][i])
with open(new_name, 'w') as f:
    f.write(fstring)
print('wrote {}'.format(new_name))

#clabels = ['optx', 'opty', 'optz']
#clabels = ['emx', 'emy', 'emz']
#plotres = s.residuals * 0.001
#xyscale = 0.001
#scale = 0.3
#
#f, a = plt.subplots(1, 3, clear=True, figsize=(12, 12), num=2)
#for row, (x, y) in enumerate(itertools.combinations([0, 1, 2], 2)):
#    #for k in range(3):
#        #sc = a[row][k].scatter(
#        #        s.solves[0].data['dst'][:, x] * xyscale,
#        #        s.solves[0].data['dst'][:, y] * xyscale,
#        #        c=plotres[:, k],
#        #        marker='o',
#        #        s=2.5,
#        #        cmap='jet',
#        #        vmin=-5,
#        #        vmax=5)
#    sc = a[row].quiver(
#            s.solves[0].data['dst'][:, x] * xyscale,
#            s.solves[0].data['dst'][:, y] * xyscale,
#            plotres[:, x],
#            plotres[:, y],
#            np.linalg.norm(plotres, axis=1) * 500,
#            angles='xy',
#            scale=scale,
#            scale_units='xy',
#            cmap='jet')
#
#    f.colorbar(sc, ax=a[row])
#    #a[row][k].plot(
#    #        s.solves[-1].transform.control_pts[:, x] * xyscale,
#    #        s.solves[-1].transform.control_pts[:, y] * xyscale,
#    #        'xk',
#    #        markersize=1.5)
#    a[row].set_xlabel(clabels[x])
#    a[row].set_ylabel(clabels[y])
#    #if row == 0:
#    #    a[row].set_title('residual_%s' % clabels[-1])
#
##f, a = plt.subplots(4, 2, num=5, clear=False)
#f = plt.figure(5)
#row = 3
#x = 0
#y = 2
#a0 = plt.subplot(4, 2, row * 2 + 1)
#a1 = plt.subplot(4, 2, row * 2 + 2)
#a0.quiver(
#        s.solves[0].data['dst'][:, x] * xyscale,
#        s.solves[0].data['dst'][:, y] * xyscale,
#        plotres[:, x],
#        plotres[:, y],
#        np.linalg.norm(plotres, axis=1) * 500,
#        angles='xy',
#        scale=scale,
#        scale_units='xy',
#        cmap='jet')
#a1.hist(rmag, bins=np.arange(0, 50, 2.0), color='g', alpha=0.5, edgecolor='k')

leave_out = False
leave_out_frac = 1.0


def solve_job(args):
    s = StagedSolve(input_data=args, args=[])
    s.run()
    res = {
            s.leave_out_label: {
                'leave_out_rmag': s.leave_out_rmag,
                'leave_out_res': s.leave_out_res,
                'avdelta_cntrl': s.avdelta,
                'leave_in_res': np.linalg.norm(s.residuals, axis=1).mean()
                }
            }
    return res


def leave_one_out(args, reg, nmax, n):
    ind = np.arange(nmax)
    np.random.shuffle(ind)
    ind = ind[0: n]

    allargs = []
    for i in ind:
        allargs.append(copy.deepcopy(args))
        allargs[-1]['leave_out_index'] = i

    pool = multiprocessing.Pool(4)
    results = pool.map(solve_job, allargs, chunksize=50)

    loo = {}
    for r in results:
        loo.update(r)

    return loo


if leave_out:
    nmax = s.solves[0].data['src'].shape[0]
    r = None
    loo = leave_one_out(args, r, nmax, int(nmax * leave_out_frac))
    
    factor = 0.001
    labels = []
    tps_displacements = []
    residuals = []
    loo_residuals = []
    for result in list(loo.items()):
        labels.append(result[0])
        tps_displacements.append(result[1]['avdelta_cntrl'])
        residuals.append(result[1]['leave_in_res'])
        loo_residuals.append(result[1]['leave_out_rmag'])
    loo_residuals = np.array(loo_residuals) * factor
    ind = np.argsort(loo_residuals)[::-1]
    loo_residuals = loo_residuals[ind]
    residuals = np.array(residuals)[ind] * factor
    labels = np.array(labels)[ind]

    plt.figure(2)
    plt.clf()
    mloo = loo_residuals[loo_residuals < 25.].mean()
    plt.axvline(mloo, color='k', linestyle='--', label="mean = %0.1fum" % mloo)
    plt.hist(loo_residuals, bins=np.arange(0, 20, 0.5), alpha=0.5, color='g', edgecolor='k')
    plt.xlabel('leave-out residual [um]', fontsize=18)
    plt.ylabel('count', fontsize=18)
    plt.title(args['data']['landmark_file'], fontsize=8)
    plt.legend(fontsize=18)

    for i in range(20):
        print("%10s %10.1fum" % (labels[i], loo_residuals[i]))

    loo_name = os.path.join(
            os.path.dirname(s.args['data']['landmark_file']),
            os.path.splitext(
                os.path.basename(s.args['data']['landmark_file']))[0] +
            '_leave_outs.json')
    with open(loo_name, 'w') as f:
        json.dump(loo, f, indent=2)
    
    #fig = plt.figure(1)
    #fig.clf()
    #ax1 = fig.add_subplot(111)
    #pres = np.linalg.norm(s.solves[0].residuals, axis=1).mean() * factor
    #ax1.axhline(pres, linestyle=':', color='b', label='polynomial')
    #ax1.plot(reg, residuals, 'b', label='polynomial + spline')
    #ax1.plot(reg, loo_residuals, '--b', label='leave-one-out\npolynomial + spline')
    #ax1.tick_params('y', colors='b')
    #ax1.set_ylabel('Residuals [um]', fontsize=14, color='b')
    #ax1.set_xlabel('Spline Regularization', fontsize=14)
    #ax1.set_xscale('log')
    #ax1.legend(loc=5, title='mean residuals (left axis)')
    #
    #ax2 = ax1.twinx()
    #ax2.plot(reg, tps_displacements, 'r')
    #ax2.tick_params('y', colors='r')
    #ax2.set_ylabel('Spline displacements [um]', fontsize=14, color='r')
