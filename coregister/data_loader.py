import numpy as np
import argschema
import pandas
import re
from .schemas import DataLoaderSchema

example1 = {
        'landmark_file': './data/17797_2Pfix_EMmoving_20190405_PA_1724_merged.csv',
        'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
        'actions': ['invert_opty'],
        'sd_set': {'src': 'opt', 'dst': 'em'}
        }

example2 = {
        'landmark_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize.csv',
        'header': ['optz', 'opty', 'optx'],
        'sd_set': {'src': 'opt', 'dst': 'em'}
        }

example3 = {
        'landmark_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize_labeled_filtered.csv',
        'header': ['label', 'optx', 'opty', 'optz'],
        'sd_set': {'src': 'opt', 'dst': 'em'}
        }


def invert_y(y):
    return 1.322 - y


def px_to_mm(x):
    return 0.002 * x


class DataLoader(argschema.ArgSchemaParser):
    """class to load and manipulate different sources of data
    """
    default_schema = DataLoaderSchema

    def run(self):
        self.df = pandas.read_csv(
                self.args['landmark_file'],
                header=None,
                names=self.args['header'])

        # make a label column if there is not one
        if 'label' not in self.df.columns:
            self.df.loc[:, 'label'] = pandas.Series(
                    np.arange(self.df.shape[0]).astype('str'))

        # if flag is present, remove data where not flag
        if (not self.args['all_flags']) & ('flag' in self.df.columns):
            self.df = self.df[self.df['flag']]

        if self.df['label'].dtype == 'int64':
            label_nums = self.df['label'].values
        else:
            label_nums = np.array([int(re.findall("\d+", lab)[0]) for lab in self.df['label']])
        ind = (label_nums > self.args['exclude_labels'][0]) & \
                  (label_nums < self.args['exclude_labels'][1])
        ind = np.invert(ind)
        self.df = self.df[ind]

        # check specific actions
        if 'invert_opty' in self.args['actions']:
            self.df['opty'] = invert_y(self.df['opty'])

        if 'opt_px_to_mm' in self.args['actions']:
            for k in self.df.columns:
                if 'opt' in k:
                    self.df[k] = px_to_mm(self.df[k])

        self.data = {}
        if self.args['all_flags']:
            self.data['flag'] = self.df['flag'].values
        self.data['labels'] = self.df['label'].values
        self.data['sd_set'] = dict(self.args['sd_set'])
        for k in ['src', 'dst']:
            a = [self.args['sd_set'][k] + xyz for xyz in ['x', 'y', 'z']]
            if set(a).issubset(set(self.df.columns)):
                self.data[k] = self.df[a].values.astype('float')


if __name__ == '__main__':  # pragma: no cover
    d1 = DataLoader(input_data=example1, args=[])
    d1.run()
    d2 = DataLoader(input_data=example2, args=[])
    d2.run()
