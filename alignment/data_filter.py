import scipy.spatial
from .data_handler import DataLoader
from .schemas import DataFilterSchema
import argschema
import numpy as np
import logging

example = {
        'dset1': {
            'landmark_file': './data/17797_2Pfix_EMmoving_20190405_PA_1724_merged.csv',
            'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
            'actions': ['invert_opty'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
            },
        'dset2': {
            'landmark_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize.csv',
            'header': ['optz', 'opty', 'optx'],
            'actions': ['opt_px_to_mm'],
            'sd_set': {'src': 'opt', 'dst': 'em'}
            },
        'output_file': './data/animal_id-17797_session-9_stack_idx-19_pixel-centroids_pre-resize_filtered.csv',
        'header': 'opt',
        }


class DataFilter(argschema.ArgSchemaParser):
    # filters out dset2 data with the convex hull of dset1
    default_schema = DataFilterSchema

    def run(self):
        d1 = DataLoader(input_data=self.args['dset1'])
        d1.run()
        self.logger.setLevel('INFO')
        self.logger.info("\nbasis data spans: {} to {}".format(
            d1.data['src'].min(axis=0),
            d1.data['src'].max(axis=0)))
        d2 = DataLoader(input_data=self.args['dset2'])
        d2.run()
        self.logger.setLevel('INFO')
        self.logger.info("\nto be filtered data spans: {} to {}".format(
            d2.data['src'].min(axis=0),
            d2.data['src'].max(axis=0)))
        self.logger.info("\nto be filtered data has shape: {}".format(
            d2.data['src'].shape))

        hull = scipy.spatial.Delaunay(d1.data['src'])
        inside = hull.find_simplex(d2.data['src']) >= 0

        newdata = d2.data['src'][inside]

        self.logger.info("\nfiltered data spans: {} to {}".format(
            newdata.min(axis=0),
            newdata.max(axis=0)))
        self.logger.info("\nfiltered data has shape: {}".format(
            newdata.shape))

        newdata[:, 1] = (661 - newdata[:, 1] / 0.002) * 0.002

        np.savetxt(self.args['output_file'], newdata, delimiter=',', fmt='%0.6f')


if __name__ == '__main__':
    dfmod = DataFilter(input_data=example, args=[])
    dfmod.run()
