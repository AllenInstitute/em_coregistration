from . polynomial import PolynomialModel
from . chunked import ChunkedModel
from . spline import SplineModel


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


# class StagedTransform():
#     def __init__(self, tflist):
#         self.tflist = tflist
#
#     def transform(self, coords):
#         x = np.copy(coords)
#         for tf in self.tflist:
#             x = tf.tform(x)
#         return x
