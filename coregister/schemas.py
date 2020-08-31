from argschema import ArgSchema
from argschema.schemas import DefaultSchema
from argschema.fields import (
        InputFile, List, Str, Dict,
        Nested, Int, OutputFile,
        OutputDir, Bool)


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
        cli_as_single_argument=True,
        missing=[100000, 200000],
        default=[100000, 200000],
        description="ignore Pt labels in this range")

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
    output_dir = OutputDir(
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
    output_dir = OutputDir(
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
