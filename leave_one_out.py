from coregister.solve import Solve3D
from coregister.data_loader import DataLoader
import multiprocessing
import numpy as np
import copy
import os
import json
import tempfile

leave_out_frac = 1.0

with open("./data/staged_transform_args.json", "r") as f:
    args = json.load(f)
#tlast = copy.deepcopy(args['transform']['transforms'][-1])
#args['transform']['transforms'] = args['transform']['transforms'][0:3]
#args['transform']['transforms'].append(tlast)

def solve_job(args):
    with tempfile.NamedTemporaryFile() as temp:
        s = Solve3D(input_data=args, args=['--output_json', temp.name])
        s.run()
    res = {
            s.left_out['labels'][0]: {
                'leave_out_rmag': s.leave_out_rmag.tolist(),
                'leave_out_res': s.leave_out_res.tolist(),
                }
            }
    print(res)
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
