import numpy as np
import argschema
import pandas
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


class DataLoader(argschema.ArgSchemaParser):
    """class to load and manipulate different sources of data
    """
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

        # check specific actions
        if 'invert_opty' in self.args['actions']:
            df['opty'] = 1.322 - df['opty']

        if 'opt_px_to_mm' in self.args['actions']:
            for k in df.columns:
                if 'opt' in k:
                    df[k] *= 0.002

        if 'em_nm_to_neurog' in self.args['actions']:
            df['emx'] = df['emx'] / 4 - 3072
            df['emy'] = df['emy'] / 4 - 2560
            df['emz'] = (df['emz']/960.) * 24 + 7924

        self.data = {}
        self.data['labels'] = df['label'].values
        self.data['sd_set'] = dict(self.args['sd_set'])
        for k in ['src', 'dst']:
            a = [self.args['sd_set'][k] + xyz for xyz in ['x', 'y', 'z']]
            if set(a).issubset(set(df.columns)):
                self.data[k] = df[a].values


if __name__ == '__main__':
    d1 = DataLoader(input_data=example1, args=[])
    d1.run()
    d2 = DataLoader(input_data=example2, args=[])
    d2.run()
