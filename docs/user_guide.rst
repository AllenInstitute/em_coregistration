User Guide
==========

example 1 in alignment/solve_3D shows how to create a transform that transforms optical data into EM coordinates. example 2 (in the code) shows how to go from optical to EM::

    example1 = {
            'data': {
                'landmark_file' : './data/17797_2Pfix_EMmoving_20190414_PA_1018_Deliverable20180415.csv',
                'header': ['label', 'flag', 'emx', 'emy', 'emz', 'optx', 'opty', 'optz'],
                'actions': ['invert_opty'],
                'sd_set': {'src': 'em', 'dst': 'opt'}
            },
            'output_json': '/allen/programs/celltypes/workgroups/em-connectomics/danielk/3D_alignment/tmp_out/transform.json',
            'model': 'TPS',
            'npts': 10,
            'regularization': {
                'translation': 1e0,
                'linear': 1e0,
                'other': 1e20
                }
    }

some explanation:

* landmark file : coming from the coregistration team. headerless csv
* header : rather than copy and paste in a header, I just specify it here.
* actions : the landmark file came from the coregistration team finding landmarks between 2 stacks in Fiji. But, the optical one was inverted in y relative to Baylor's original coordinates. So, flip it. In the code, this is really hard-coded to the particular dimensions from this volume.
* sd_set : what is the source (moving) and what is the destination (fixed)
* output_json : serialized output of the resulting transform
* model : specifying Thin Plate Spline
* npts : resulting spline will have npts^3 control points
* regularization : a way to keep things under control. We want the affine components of the transform to take care of big rotations and scalings and the *other* to be relatively small tweaks on top of that.


