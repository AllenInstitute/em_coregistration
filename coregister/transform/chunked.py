import numpy as np
from . polynomial import PolynomialModel


class ChunkedModel():
    def __init__(
            self, axis=2, json=None, parameters=None, regularization=0,
            order=1, nchunks=None, range_min=None, range_max=None,
            ranges=None, transforms=None):

        if json is not None:
            self.from_dict(json)
            return

        self.nchunks = nchunks
        self.axis = axis
        self.transforms = transforms
        self.set_ranges(nchunks=nchunks, range_min=range_min, ranges=ranges)
        if self.transforms is None:
            self.transforms = []
            for i in range(self.nchunks):
                self.transforms.append(
                        PolynomialModel(
                            regularization=regularization,
                            order=order))

    def from_dict(self, json):
        self.nchunks = json['nchunks']
        self.transforms = [PolynomialModel(json=j) for j in json['transforms']]
        self.axis = json['axis']
        if 'ranges' in json:
            self.ranges = np.array(json['ranges'])

    def to_dict(self):
        json = {
                'name': 'ChunkedModel',
                'nchunks': self.nchunks,
                'axis': self.axis,
                'transforms': [t.to_dict() for t in self.transforms]
                }
        if hasattr(self, 'ranges'):
            json['ranges'] = self.ranges.tolist()
        return json

    def set_ranges(
            self, nchunks=None, range_min=None, range_max=None, ranges=None):
        if (ranges is None) & (range_min is None):
            # do not set anything
            return
        if ranges is not None:
            self.ranges = ranges
        else:
            self.ranges = np.linspace(range_min, range_max, nchunks)[1:-1]
        self.nchunks = self.ranges.size + 1

    def set_ranges_from_src(self, src, axis, nchunks):
        self.axis = axis
        self.set_ranges(
                nchunks=nchunks,
                range_min=src[:, axis].min(),
                range_max=src[:, axis].max())

    def tform(self, src):
        which = np.searchsorted(self.ranges, src[:, self.axis])
        dst = np.zeros_like(src)
        for i in range(self.nchunks):
            inds = which == i
            dst[inds, :] = self.transforms[i].tform(src[inds, :])
        return dst

    def estimate(self, src, dst):
        if not hasattr(self, 'ranges'):
            self.set_ranges_from_src(src, self.axis, self.nchunks)
        which = np.searchsorted(self.ranges, src[:, self.axis])
        for i in range(self.nchunks):
            inds = which == i
            self.transforms[i].estimate(src[inds, :], dst[inds, :])
