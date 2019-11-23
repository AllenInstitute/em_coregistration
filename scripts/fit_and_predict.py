from alignment.staged_solve import StagedSolve
from alignment.data_handler import invert_y
import copy
import os
import json


with open("./data/staged_transform_args.json", "r") as f:
    args = json.load(f)


s = StagedSolve(input_data=copy.deepcopy(args), args=[])
s.run()

rlist = s.sorted_labeled_residuals()
for r in rlist[0:10]:
    print("%10s %10.1f" % (r[0], r[1]))

alldata = s.predict_all_data()
# re-invert y for output back to annotators
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
