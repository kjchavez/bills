import pprint

def _recursive_keys(x):
    keys = {}
    for key in x:
        if isinstance(x[key], dict):
            keys[key] = _recursive_keys(x[key])
        elif isinstance(x[key], str):
            keys[key] = x[key][0:10] + ("..." if len(x[key]) > 10 else "")
        else:
            keys[key] = str(type(x[key]))

    return keys

def print_schema(example, file=None):
    """ Prints a nicely formatted truncated schemata for |example|. """
    pprint.pprint(_recursive_keys(example), stream=file)
