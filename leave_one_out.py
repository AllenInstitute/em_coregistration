from coregister.staged_solve import StagedSolve
from coregister.data_handler import DataLoader
import multiprocessing
import numpy as np
import copy
import os
import json

leave_out_frac = 1.0

with open("./data/staged_transform_args.json", "r") as f:
    args = json.load(f)


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


def leave_one_out(args, nmax, n):
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


data = DataLoader(input_data=copy.deepcopy(args['data']), args=[])
data.run()
nmax = data.data['src'].shape[0]
loo = leave_one_out(args, nmax, int(nmax * leave_out_frac))

loo_name = os.path.join(
        os.path.dirname(args['data']['landmark_file']),
        os.path.splitext(
            os.path.basename(args['data']['landmark_file']))[0] +
        '_leave_outs.json')
with open(loo_name, 'w') as f:
    json.dump(loo, f, indent=2)
