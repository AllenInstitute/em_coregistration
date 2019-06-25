import numpy as np
import scipy.spatial


def linear_kernel(src, control_pts=None):
    A = np.zeros((src.shape[0], 4))
    A[:, 0] = 1.0 # offset
    A[:, 1:4] = src
    return A


def polynomial_kernel(src, control_pts=None):
    A = np.zeros((src.shape[0], 10))
    A[:, 0] = 1.0 # offset
    A[:, 1:4] = src
    A[:, 4] = src[:, 0]**2 # x^2
    A[:, 5] = src[:, 1]**2 # y^2
    A[:, 6] = src[:, 2]**2 # z^2
    A[:, 7] = src[:, 0] * src[:, 1] # xy
    A[:, 8] = src[:, 0] * src[:, 2] # xz
    A[:, 9] = src[:, 1] * src[:, 2] # yz
    return A


def thin_plate_kernel(src, control_pts=None):
    A = np.zeros((src.shape[0], 4 + control_pts.shape[0]))
    A[:, 0] = 1.0 # offset
    A[:, 1:4] = src # affine
    disp = scipy.spatial.distance.cdist(
            src,
            control_pts,
            metric='sqeuclidean')
    disp *= np.ma.log(np.sqrt(disp)).filled(0.0)
    A[:, 4:] = disp
    return A


class Transform():

    def __init__(self, model, control_pts=None):
        self.control_pts = control_pts
        self.model = model

    def kernel(self, src):
        if self.model == 'LIN':
            return linear_kernel(src)
        elif self.model == 'POLY':
            return polynomial_kernel(src)
        elif self.model == 'TPS':
            return thin_plate_kernel(src, self.control_pts)

    def load_parameters(self, x):
        self.parameters = x

    def transform(self, src):
        k = self.kernel(src)
        dst = np.zeros_like(src)
        for i in range(src.shape[1]):
            dst[:, i] = k.dot(self.parameters[:, i])
        return dst

    def to_dict(self):
        return {
                'model': self.model,
                'parameters': self.parameters.tolist(),
                'control_pts': self.control_pts.tolist()
                }

    def from_dict(self, j):
        self.model = j['model']
        self.parameters = np.array(j['parameters'])
        self.control_pts = np.array(j['control_pts'])
