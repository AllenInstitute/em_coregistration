import numpy as np
import argschema
import pandas
from .schemas import DataLoaderSchema

example1 = {
        'landmark_file': './data/17797_2Pfix_EMmoving_20190405_PA_1724_merged.csv',
        'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
        'actions': {'opty': 'invert'},
        'sd_set': {'opt': 'src', 'em': 'dst'}
        }

example2 = {
        'landmark_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize.csv',
        'header': ['optz', 'opty', 'optx'],
        'sd_set': {'opt': 'src', 'em': 'dst'}
        }


class DataLoader(argschema.ArgSchemaParser):
    default_schema = DataLoaderSchema

    def run(self):
        df = pandas.read_csv(
                self.args['landmark_file'],
                header=None,
                names=self.args['header'])

        # make a label column if there is not one
        if 'label' not in df.columns:
            df.loc[:, 'label'] = pandas.Series(
                    np.arange(df.shape[0]).astype('str'))

        # if flag is present, remove data where not flag
        if 'flag' in df.columns:
            df = df[df['flag']]

        # invert y if specified
        for k, v in self.args['actions'].items():
            if (k == 'opty') & (v == 'invert'):
                df[k] = (661 - df[k] / 0.002) * 0.002

        self.data = {}
        self.data['labels'] = df['label'].values
        self.data['sd_set'] = self.args['sd_set']
        for k in ['opt', 'em']:
            a = [k + xyz for xyz in ['x', 'y', 'z']]
            if set(a).issubset(set(df.columns)):
                self.data[self.args['sd_set'][k]] = df[a].values


if __name__ == '__main__':
    d1 = DataLoader(input_data=example1, args=[])
    d1.run()
    d2 = DataLoader(input_data=example2, args=[])
    d2.run()
