"""Automatically generate the Python API documentation pages by parsing the public
functions from `__all__`.
"""

import re
from pathlib import Path

import mkdocs_gen_files

# dictionary structure
# key: python_api directory path
# value: (paths to include, namespace)
doc_sections = {
    'integrators': (['dynamiqs/integrators/'], 'dq'),
    'qarrays/qarray': (['dynamiqs/qarrays/qarray.py'], 'dq'),
    'time_qarray': (['dynamiqs/time_qarray.py'], 'dq'),
    'method': (['dynamiqs/method.py'], 'dq.method'),
    'gradient': (['dynamiqs/gradient.py'], 'dq.gradient'),
    'hermitian_conjugate': (['dynamiqs/hermitian_conjugate.py'], 'dq'),
    'utils/operators': (['dynamiqs/utils/operators.py'], 'dq'),
    'utils/states': (['dynamiqs/utils/states.py'], 'dq'),
    'utils/general': (
        ['dynamiqs/utils/general.py', 'dynamiqs/utils/wigner_utils.py'],
        'dq',
    ),
    'qarrays/utils': (['dynamiqs/qarrays/utils.py'], 'dq'),
    'utils/global_settings': (['dynamiqs/utils/global_settings.py'], 'dq'),
    'utils/vectorization': (['dynamiqs/utils/vectorization.py'], 'dq'),
    'utils/optimal_control': (['dynamiqs/utils/optimal_control.py'], 'dq'),
    'random': (['dynamiqs/random/'], 'dq.random'),
    'plot': (['dynamiqs/plot/'], 'dq.plot'),
}


def get_elements_from_all(file_path):
    """Parse a file to find all elements of the `__all__` attribute."""
    with Path.open(file_path) as f:
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


# generate a documentation file for each function of each file/directory
for section, (paths, namespace) in doc_sections.items():
    # e.g 'python_api/operators' or 'python_api/plot'
    doc_path = Path('python_api', section)

    for path in paths:
        # start with e.g. 'dynamiqs/utils/operators.py' or 'dynamiqs/plot'
        src_path = Path(path)

        # convert to e.g 'dynamiqs.utils.operators' or 'dynamiqs.plot'
        identifier = src_path.with_suffix('').as_posix().replace('/', '.')

        if src_path.is_dir():
            src_path = src_path / '__init__.py'

        # loop over all functions in file
        for function in get_elements_from_all(src_path):
            # convert to e.g 'python_api/utils/operators/eye.md'
            doc_path_function = Path(doc_path, function).with_suffix('.md')

            # create the function page
            with mkdocs_gen_files.open(doc_path_function, 'w') as f:
                module = identifier.split('.')[0]
                print(f'::: {identifier}.{function}', file=f)
                print('    options:', file=f)
                print(f'        namespace: {namespace}', file=f)

            mkdocs_gen_files.set_edit_path(doc_path_function, Path('..') / src_path)
