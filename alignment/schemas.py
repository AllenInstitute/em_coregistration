from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        InputFile, List, Str, Dict,
        Nested, Int, Float, OutputFile,
        OutputDir)


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


class SolverSchema(ArgSchema):
    data = Nested(DataLoaderSchema)
    regularization = Nested(regularization)
    leave_out_index = Int(
        required=False,
        missing=None,
        default=None,
        description="index to leave out of data")
    model = Str(
        required=False,
        default='TPS',
        missing='TPS',
        description=("LIN, POLY, or TPS for linear, polynomial, "
                     "thin plate spline"))
    npts = Int(
        required=False,
        missing=None,
        default=None,
        description="number of pts per axis for TPS controls")
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
