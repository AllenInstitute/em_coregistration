from argschema import ArgSchema
from argschema.fields import \
        InputFile, List, Str, Dict


class DataLoaderSchema(ArgSchema):
    landmark_file = InputFile(
        required=True,
        description=("csv file, one line per landmark"))
    actions = Dict(
        required=False,
        missing={},
        default={},
        description=("actions to perform on data"))
    header = List(
        Str,
        required=False,
        default=None,
        missing=None,
        cli_as_single_argument=True,
        description=("passed as names=header to pandas.read_csv()"))
    sd_set = Dict(
        required=True,
        default={'opt': 'src', 'em': 'dst'})
