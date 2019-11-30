from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        InputFile, List, Str, Dict,
        Nested, Int, Float, OutputFile,
        OutputDir, Bool)
import marshmallow as mm


class src_dst(DefaultSchema):
    src = Str(
        required=False,
        missing='opt',
        default='opt',
        description=("is optical data src or dst"))
    dst = Str(
        required=False,
        missing='em',
        default='em',
        description=("is em data src or dst"))


class DataLoaderSchema(ArgSchema):
    landmark_file = InputFile(
        required=True,
        description=("csv file, one line per landmark"))
    actions = List(
        Str,
        required=False,
        missing=[],
        default=[],
        cli_as_single_argument=True,
        description=("actions to perform on data"))
    header = List(
        Str,
        required=False,
        default=None,
        missing=None,
        cli_as_single_argument=True,
        description=("passed as names=header to pandas.read_csv()"))
    sd_set = Nested(src_dst)
    all_flags = Bool(
        required=False,
        missing=False,
        default=False,
        description="if False, returns only flag=True data")
    exclude_labels = List(
        Int,
        required=True,
        missing = [100000,200000],
        default = [100000,200000],
        description = "ignore Pt labels in this range")
    #exclude_ranges = Dict(
    #    required=False
    #    missing=None,
    #    default=None,
    #    description=("dict like {"src": {0: [a, b]}, {1, [c, d]}, ...} "
    #                 "to specify inclusive ranges"))


class regularization(ArgSchema):
    translation = Float(
        default=1e-10,
        description='regularization factor for translation')
    linear = Float(
        default=1.0,
        description='regularization factor for linear components')
    other = Float(
        default=1e5,
        description='regularization factor for everything else')
    tps = List(
        Float,
        required=False,
        default=[1e10, 1e10, 1e10],
        missing=[1e10, 1e10, 1e10],
        description='regularization factor for thin plate deformations')
    @mm.pre_load
    def tolist(self, data):
        if 'tps' in data:
            if not isinstance(data['tps'], list):
                data['tps'] = [data['tps']] * 3


#class TransformSchema(DefaultSchema):
#    regularization = Nested(regularization)
#    name = Str(
#        required=True,
#        validate=mm.validate.OneOf("AffineModel"),
#        description="name of transform model")
#    bounds_buffer = Float(
#        required=False,
#        missing=0.0,
#        default=0.0,
#        description="extend boundaries by this much for computing control points")
#    model = Str(
#        required=False,
#        default='TPS',
#        missing='TPS',
#        description=("LIN, POLY, or TPS for linear, polynomial, "
#                     "thin plate spline"))
#    npts = List(
#        Int,
#        required=False,
#        missing=None,
#        default=None,
#        description="number of pts per axis for TPS controls")
#    nz = Int(
#        required=False,
#        missing=21,
#        deault=21,
#        description="number of z slabs for chunked z")
#    axis = Str(
#        required=False,
#        missing='z',
#        default='z',
#        description='axis for chunked affine')
#
#    @mm.pre_load
#    def tolist(self, data):
#        if 'npts' in data:
#            if not isinstance(data['npts'], list):
#                data['npts'] = [data['npts']] * 3


class SolverSchema(ArgSchema):
    data = Nested(DataLoaderSchema)
    transform = Dict(
        required=True,
        description="dict containing transform specification")
    leave_out_index = Int(
        required=False,
        missing=None,
        default=None,
        description="index to leave out of data")
    output_dir= OutputDir(
        required=False,
        missing=None,
        default=None,
        description="path for writing output json of transform")


class StagedSolveSchema(ArgSchema):
    data = Nested(DataLoaderSchema)
    transforms = List(
        Dict,
        required=True,
        description="list of transform arg dicts")
    leave_out_index = Int(
        required=False,
        missing=None,
        default=None,
        description="index to leave out of data")
    output_dir= OutputDir(
        required=False,
        missing=None,
        default=None,
        description="path for writing output json of transform")


class DataFilterSchema(ArgSchema):
    dset1 = Nested(DataLoaderSchema)
    dset_soma = Nested(DataLoaderSchema)
    dset2 = Nested(DataLoaderSchema)
    output_file = OutputFile(
        required=False,
        missing=None,
        default=None,
        description="where to write output file")
    header = Str(
        required=True,
        default="opt",
        description="specifies which data to use, i.e. opt/em")
