from . polynomial import PolynomialModel
from . chunked import ChunkedModel
from . spline import SplineModel
import numpy as np


class Transform():
    """class for creating solver matrix kernel and
       performing transformations. includes serialization
       and desrialization functions.
    """

    def __init__(self, name=None, json=None, **kwargs):
        """Initialize Transform
        Parameters
        ----------
        name : str
            classname of this transform
        json : dict
            json compatible representation of this transform
            (supersedes className, dataString, and transformId if not None)
        """
        classes = {
                "PolynomialModel": PolynomialModel,
                "ChunkedModel": ChunkedModel,
                "SplineModel": SplineModel
                }

        if json is not None:
            self.__class__ = classes[json['name']]
            self.__class__.__init__(self, json=json)

        elif name is not None:
            self.__class__ = classes[name]
            self.__class__.__init__(self, **kwargs)


class TransformList():
    def __init__(self, json=None, transforms=None):
        if json is not None:
            self.from_dict(json)
            return

        if transforms is not None:
            self.transforms = [Transform(**tf) for tf in transforms]

    def tform(self, src):
        dst = np.copy(src)
        for tf in self.transforms:
            dst = tf.tform(dst)
        return dst

    def to_dict(self):
        j = [t.to_dict() for t in self.transforms]
        return j

    def from_dict(self, json):
        self.transforms = [Transform(json=j) for j in json]

    def estimate(self, src, dst):
        nsrc = np.copy(src)
        for tf in self.transforms:
            tf.estimate(nsrc, dst)
            nsrc = tf.tform(nsrc)
