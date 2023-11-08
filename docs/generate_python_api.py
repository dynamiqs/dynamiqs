"""Automatically generate the Python API documentation pages by parsing the public
functions from `__all__`."""

import re
from pathlib import Path

import mkdocs_gen_files

PATHS_TO_PARSE = [
    'dynamiqs/utils/operators.py',
    'dynamiqs/utils/states.py',
    'dynamiqs/utils/utils.py',
    'dynamiqs/utils/tensor_types.py',
    'dynamiqs/utils/wigners.py',
    'dynamiqs/utils/vectorization.py',
    'dynamiqs/utils/optimal_control.py',
    'dynamiqs/plots/misc.py',
]


def get_elements_from_all(file_path):
    """Parse a file to find all elements of the `__all__` attribute."""
    with open(file_path, 'r') as f:
        contents = f.read()

        # capture list assigned to __all__ with a regular expression (the `[^\]]+` part
        # of the regex matches one or more characters that are not the closing bracket)
        all_regex = r'__all__ = \[([^\]]+)\]'
        match = re.search(all_regex, contents, re.DOTALL)  # re.DOTALL matches newlines
        if match:
            # extract first group from the match (the part inside the brackets)
            all_str = match.group(1)
            # remove all whitespaces, newlines and single or double quotes
            all_str = all_str.translate({ord(c): None for c in ' \'"\n'})
            # strip the trailing comma (for multiline __all__ definitions) and split
            return all_str.strip(',').split(',')
        else:
            return []


# generate a documentation file for each function of each file
for path in PATHS_TO_PARSE:
    # start with e.g. 'dynamiqs/utils/operators.py'
    src_path = Path(path)
    # convert to e.g 'python_api/utils/operators'
    doc_path = Path('python_api', *src_path.parts[1:]).with_suffix('')
    # convert to e.g 'dynamiqs.utils.operators'
    identifier = src_path.with_suffix('').as_posix().replace('/', '.')

    # loop over all functions in file
    for function in get_elements_from_all(src_path):
        # convert to e.g 'python_api/utils/operators/eye.md'
        doc_path_function = Path(doc_path, function).with_suffix('.md')

        # create the function page
        with mkdocs_gen_files.open(doc_path_function, 'w') as f:
            module = identifier.split('.')[0]
            print(f'::: {identifier}.{function}', file=f)

        mkdocs_gen_files.set_edit_path(doc_path_function, Path('..') / src_path)
