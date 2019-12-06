import coregister.data_filter as df
import tempfile
import copy
import os


def test_data_filter():
    with tempfile.NamedTemporaryFile() as temp:
        args = copy.deepcopy(df.example)
        args['output_file'] = temp.name
        f = df.DataFilter(input_data=args, args=[])
        f.run()
        assert os.path.isfile(temp.name)
