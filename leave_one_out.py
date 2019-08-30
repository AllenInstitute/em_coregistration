import alignment.solve_3d as s3
import numpy as np
import copy
import multiprocessing

def job(fargs):
    i, nc = fargs
    s = s3.Solve3D(
            input_data=copy.deepcopy(s3.example2),
            args=['--leave_out_index', '%d' % i, '--npts', '%d' % nc]) 
    s.run()
    res = s.left_out['dst'] - s.transform.transform(s.left_out['src'])
    return s.left_out['labels'][0], np.linalg.norm(res[0]), np.linalg.norm(s.residuals, axis=1).mean()

leave_out_res = {}

allres = []
allk = [5]
lores = []
for k in allk:
    args = [(i, k) for i in range(1736)]
    pool = multiprocessing.Pool(8)
    results = []
    for r in pool.imap(job, args, chunksize=100):
        results.append(r)
    allres.append(np.array([i[2] for i in results]).mean())
    lores.append(np.array([i[1] for i in results]).mean())

