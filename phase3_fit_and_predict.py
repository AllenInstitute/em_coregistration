from coregister.solve import Solve3D
from coregister.data_loader import invert_y
from coregister.transform import Transform
import copy
import os
import json
import numpy as np


# solve with some inputs
#args_path = "./data/inverse_staged_transform_args.json"
#output_path = "./data/inverse_staged_transform_solution.json"
args_path = "./data/phase3_staged_transform_args.json"
output_path = "./data/phase3_staged_transform_solution.json"
s = Solve3D(args=[
    "--input_json", args_path,
    "--output_json", output_path])
s.run()
print('worst points')
for r in s.sorted_labeled_residuals[0:20]:
    print("%10s %10.6f" % (r[0], r[1]))

# look at residuals at every step of transform
with open(output_path, 'r') as f:
    jout = json.load(f)
ntransforms = len(jout['transforms'])
for i in range(1, ntransforms + 1):
    jtemp = copy.deepcopy(jout)
    jtemp['transforms'] = jtemp['transforms'][0:i]
    tf = Transform(json=jtemp)
    residuals = tf.tform(s.data['src']) - s.data['dst']
    rmag = np.linalg.norm(residuals, axis=1).mean()
    lastone = tf.transforms[-1].__class__.__name__
    infostr = "{} residual {:0.6f}".format(lastone, rmag)
    if lastone == "SplineModel":
        src = tf.transforms[-1].control_pts
        dst = tf.transforms[-1].tform(src)
        mov = np.linalg.norm(dst - src, axis=1).mean()
        infostr += " {} cntrls moved {:0.6f}".format(src.shape[0], mov)
    print(infostr)

alldata = s.predict_all_data()
# re-invert y for output back to annotators
#alldata.data['src'][:, 1] = invert_y(alldata.data['src'][:, 1])

# write an updated file
new_name = os.path.join(
        os.path.dirname(s.args['data']['landmark_file']),
        os.path.splitext(
            os.path.basename(s.args['data']['landmark_file']))[0] +
        '_updated.csv')
fmt = '%s,%s,%0.1f,%0.1f,%0.1f,%0.6f,%0.6f,%0.6f\n'
#fmt = '"%s","%s","%0.1f","%0.1f","%0.1f","%0.6f","%0.6f","%0.6f"\n'
fstring = ""
for i in range(alldata.data['src'].shape[0]):
    fstring += fmt % (
       alldata.data['labels'][i],
       alldata.data['flag'][i],
       *alldata.data['src'][i],
       *alldata.data['dst'][i])
with open(new_name, 'w') as f:
    f.write(fstring)
print('wrote {}'.format(new_name))
