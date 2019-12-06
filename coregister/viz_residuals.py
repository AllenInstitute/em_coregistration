import argschema
import marshmallow as mm
import numpy as np
import matplotlib.pyplot as plt
import itertools


class VizResidualsSchema(argschema.ArgSchema):
    residuals = argschema.fields.List(
        argschema.fields.List(argschema.fields.Float),
        required=True,
        description=("[[dx1, dy1, dz1], [dx2, dy2, dz2], ...]"))
    positions = argschema.fields.List(
        argschema.fields.List(argschema.fields.Float),
        required=True,
        description=("[[x1, y1, z1], [x2, y2, z2], ...]"))
    arrow_scale = argschema.fields.Float(
        required=False,
        missing=10.0,
        default=10.0,
        description="scale of arrows in quiver plot")


    @mm.post_load
    def check_dims(self, data):
        lres = np.array([len(i) for i in data['residuals']])
        lpos = np.array([len(i) for i in data['positions']])
        if (not np.all(lres == 3)) | (not np.all(lpos == 3)):
            raise mm.ValidationError("all list entries should be length 3")
        if lres.size != lpos.size:
            raise mm.ValidationError("residuals and positions should be same length")

        data['residuals'] = np.array(data['residuals'])
        data['positions'] = np.array(data['positions'])


def make_panel(axis, xpos, ypos, xres, yres, mag, xlabel, ylabel, scale):
    axis.quiver(
            xpos, ypos, xres, yres, mag,
            angles='xy', scale=1.0 / scale, scale_units='xy')
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


class VizResiduals(argschema.ArgSchemaParser):
    default_schema = VizResidualsSchema

    def run(self):
        rmag = np.linalg.norm(self.args['residuals'], axis=1)
        labels = ['x', 'y', 'z']
        f, a = plt.subplots(1, 3, clear=True, num=1)
        for i, inds in enumerate(itertools.combinations([0, 1, 2], 2)):
            make_panel(
                    a[i],
                    self.args['positions'][:, inds[0]],
                    self.args['positions'][:, inds[1]],
                    self.args['residuals'][:, inds[0]],
                    self.args['residuals'][:, inds[1]],
                    rmag,
                    labels[inds[0]],
                    labels[inds[1]],
                    self.args['arrow_scale'])


if __name__ == "__main__":
    vmod = VizResiduals()
    vmod.run()

