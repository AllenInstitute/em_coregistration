import staged_solve as s
import numpy as np
import copy
import multiprocessing
import os


def job(fargs):
    i, nc = fargs
    #s = s3.Solve3D(
    #        input_data=copy.deepcopy(s3.example2),
    #        args=['--leave_out_index', '%d' % i, '--npts', '%d' % nc]) 
    #s.run()
    #res = s.left_out['dst'] - s.transform.transform(s.left_out['src'])
    args_poly = copy.deepcopy(s3.example1)
    args_poly['model'] = 'POLY'
    s_poly = s3.Solve3D(input_data=args_poly, args=['--leave_out_index', '%d' % i])
    s_poly.run()
    tf_poly = s_poly.transform
    rng = np.random.RandomState(i)
    d = rng.randint(0, 1e6)
    tmp_path = './tmp_%d.csv' % d
    s3.write_src_dst_to_file(
            tmp_path,
            tf_poly.transform(s_poly.data['src']),
            s_poly.data['dst'])

    args_tps = copy.deepcopy(s3.example1)
    args_tps['model'] = 'TPS'
    args_tps['npts'] = nc
    args_tps['data'] = {
            'landmark_file': tmp_path,
            'header': ['polyx', 'polyy', 'polyz', 'optx', 'opty', 'optz'],
            'sd_set': {'src': 'poly', 'dst': 'opt'}
            }
    args_tps['regularization']['other'] = 1e-5
    s_tps = s3.Solve3D(input_data=args_tps, args=[])
    s_tps.run()
    tf_tps = s_tps.transform
    tf_total = StagedTransform([tf_poly, tf_tps])

    res = s_poly.left_out['dst'] - tf_total.transform(s_poly.left_out['src'])
    os.remove(tmp_path)
    return s_poly.left_out['labels'][0], np.linalg.norm(res[0]), np.linalg.norm(s_tps.residuals, axis=1).mean()

leave_out_res = {}

allres = []
allk = [10]
lores = []
for k in allk:
    args = [(i, k) for i in range(1736)]
    #args = [(i, k) for i in range(100)]
    pool = multiprocessing.Pool(8)
    results = []
    for r in pool.imap(job, args, chunksize=100):
        results.append(r)
    allres.append(np.array([i[2] for i in results]).mean())
    lores.append(np.array([i[1] for i in results]).mean())

