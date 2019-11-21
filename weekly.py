from alignment.staged_solve import StagedSolve
from alignment.data_handler import DataLoader, invert_y
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import itertools
import multiprocessing
import json

args = {
        'data': {
            "landmark_file": "data/17797_2Pfix_EMmoving_20191010_1652_piecewise_trial_updated_Master.csv",
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty', 'em_nm_to_neurog'],
            'sd_set': {'src': 'opt', 'dst': 'em'},
            'exclude_labels': [60000, 60001]
        },
        'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/em_coregistration/tmp_out/transform.json',
        "transforms": [
            {
                'model': 'POLY',
                'npts': 10,
                'regularization': {
                    'translation': 1e-15,
                    'linear': 1e-10,
                    'other': 1e-2,
                    }
            },
            {
                'model': 'ZCHUNKED',
                'bounds_buffer': 1.0,
                'npts': [5, 5, 20],
                'regularization': {
                    'translation': 1e-10,
                    'linear': 1e-10,
                    'other': 1e-10,
                    },
                "nz": 5,
                "axis": "z"
            },
            {
                'model': 'ZCHUNKED',
                'bounds_buffer': 1.0,
                'npts': [5, 5, 20],
                'regularization': {
                    'translation': 1e-10,
                    'linear': 1e-10,
                    'other': 1e-10,
                    },
                "nz": 21,
                "axis": "z"
            },
            #{
            #    'model': 'ZCHUNKED',
            #    'bounds_buffer': 1.0,
            #    'npts': [5, 5, 20],
            #    'regularization': {
            #        'translation': 1e-10,
            #        'linear': 1e-10,
            #        'other': 1e-10,
            #        },
            #    "nz": 21,
            #    "axis": "x"
            #},
            #{
            #    'model': 'ZCHUNKED',
            #    'bounds_buffer': 1.0,
            #    'npts': [5, 5, 20],
            #    'regularization': {
            #        'translation': 1e-10,
            #        'linear': 1e-10,
            #        'other': 1e-10,
            #        },
            #    "nz": 21,
            #    "axis": "y"
            #},
            #{
            #    'model': 'PIECEWISE',
            #    'npts': [3, 3, 11],
            #    'regularization': {
            #        'translation': 1e-6,
            #        'linear': 1e-6,
            #        'other': 1e-6,
            #        }
            #}
            {
                'model': 'TPS',
                'npts': [3, 3, 3],
                'regularization': {
                    'translation': 1e-6,
                    'linear': 1e-6,
                    'other': 1e10,
                    }
            },
            {
                'model': 'TPS',
                'npts': [5, 5, 5],
                'regularization': {
                    'translation': 1e-6,
                    'linear': 1e-6,
                    'other': 1e10,
                    }
            },
            #{
            #    'model': 'TPS',
            #    'npts': [6, 6, 6],
            #    'regularization': {
            #        'translation': 1e-6,
            #        'linear': 1e-6,
            #        'other': 1e10,
            #        }
            #},
            {
                'model': 'TPS',
                'npts': [10, 10, 10],
                'regularization': {
                    'translation': 1e-6,
                    'linear': 1e-6,
                    'other': 1e10,
                    }
            },
            #{
            #    'model': 'TPS',
            #    'npts': [9, 9, 9],
            #    'regularization': {
            #        'translation': 1e-6,
            #        'linear': 1e-6,
            #        'other': 1e10,
            #        }
            #},
            {
                'model': 'TPS',
                'npts': [12, 12, 12],
                'regularization': {
                    'translation': 1e-6,
                    'linear': 1e-6,
                    'other': 1e10,
                    }
            },
            {
                'model': 'TPS',
                'npts': [-1, -1, -1],
                'regularization': {
                    'translation': 1e-6,
                    'linear': 1e-6,
                    'other': 1e5,
                    }
            },
            ]
        }

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

s = StagedSolve(input_data=copy.deepcopy(args), args=[])
s.run()

# print worst residuals
rmag = np.linalg.norm(s.residuals, axis=1) * 0.001
ind = np.argsort(rmag)[::-1]
print('worst residuals:')
for i in ind[0:10]:
    print('%10s %10.1fum' % (s.solves[0].data['labels'][i], rmag[i]))

# read in all the data
alldata = DataLoader(input_data=copy.deepcopy(args['data']), args=['--all_flags', 'True'])
alldata.run()
# predict just for the False flags
inds = alldata.data['flag'] == False
pred_src = alldata.data['src'][inds]
pred_dst = s.transform.transform(pred_src)

# how much did these predictions move?
dst = alldata.data['dst'][inds]
labels = alldata.data['labels'][inds]
delta = np.linalg.norm(dst - pred_dst, axis=1) * 0.001

f, a = plt.subplots(1, 1, clear=True, figsize=(12, 4), num=1)
a.hist(delta, bins=np.logspace(-2, 3, 100), color='g', alpha=0.5, edgecolor='k')
a.set_yscale('log')
a.set_xscale('log')
a.legend()
a.set_xlabel('Change in prediction [um]', fontsize=18)
a.set_ylabel('point count', fontsize=18)

wind = np.argsort(delta)[::-1]
for i in wind[0:20]:
    print(labels[i], delta[i])

# reinvert the data
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
    if alldata.data['flag'][i]:
        fstring += fmt % (
           alldata.data['labels'][i],
           alldata.data['flag'][i],
           *alldata.data['dst'][i],
           *alldata.data['src'][i])
    else:
        pind = np.argwhere(labels == alldata.data['labels'][i]).flatten()[0]
        fstring += fmt % (
           alldata.data['labels'][i],
           alldata.data['flag'][i],
           *pred_dst[pind],
           *alldata.data['src'][i])
with open(new_name, 'w') as f:
    f.write(fstring)
print('wrote {}'.format(new_name))

clabels = ['optx', 'opty', 'optz']
clabels = ['emx', 'emy', 'emz']
plotres = s.residuals * 0.001
xyscale = 0.001
scale = 0.3

f, a = plt.subplots(1, 3, clear=True, figsize=(12, 12), num=2)
for row, (x, y) in enumerate(itertools.combinations([0, 1, 2], 2)):
    #for k in range(3):
        #sc = a[row][k].scatter(
        #        s.solves[0].data['dst'][:, x] * xyscale,
        #        s.solves[0].data['dst'][:, y] * xyscale,
        #        c=plotres[:, k],
        #        marker='o',
        #        s=2.5,
        #        cmap='jet',
        #        vmin=-5,
        #        vmax=5)
    sc = a[row].quiver(
            s.solves[0].data['dst'][:, x] * xyscale,
            s.solves[0].data['dst'][:, y] * xyscale,
            plotres[:, x],
            plotres[:, y],
            np.linalg.norm(plotres, axis=1) * 500,
            angles='xy',
            scale=scale,
            scale_units='xy',
            cmap='jet')

    f.colorbar(sc, ax=a[row])
    #a[row][k].plot(
    #        s.solves[-1].transform.control_pts[:, x] * xyscale,
    #        s.solves[-1].transform.control_pts[:, y] * xyscale,
    #        'xk',
    #        markersize=1.5)
    a[row].set_xlabel(clabels[x])
    a[row].set_ylabel(clabels[y])
    #if row == 0:
    #    a[row].set_title('residual_%s' % clabels[-1])

#f, a = plt.subplots(4, 2, num=5, clear=False)
f = plt.figure(5)
row = 3
x = 0
y = 2
a0 = plt.subplot(4, 2, row * 2 + 1)
a1 = plt.subplot(4, 2, row * 2 + 2)
a0.quiver(
        s.solves[0].data['dst'][:, x] * xyscale,
        s.solves[0].data['dst'][:, y] * xyscale,
        plotres[:, x],
        plotres[:, y],
        np.linalg.norm(plotres, axis=1) * 500,
        angles='xy',
        scale=scale,
        scale_units='xy',
        cmap='jet')
a1.hist(rmag, bins=np.arange(0, 50, 2.0), color='g', alpha=0.5, edgecolor='k')


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
