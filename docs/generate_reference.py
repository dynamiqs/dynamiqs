"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

# get navigation dictionary
nav = mkdocs_gen_files.Nav()

nav[["Solvers"]] = "solvers.md"
nav[["Solvers", "Schrödinger equation"]] = "solvers/sesolve.md"
nav[["Solvers", "Master equation"]] = "solvers/mesolve.md"
nav[["Solvers", "Stochastic master equation"]] = "solvers/smesolve.md"
nav[["Utilities"]] = "utils.md"

FILES_TO_PARSE = [
    # Path in navigation, path to reference, path to source
    (["Utilities", "Operators"], "utils/operators", "dynamiqs/utils/operators.py"),
    (["Utilities", "Quantum states"], "utils/states", "dynamiqs/utils/states.py"),
    (["Utilities", "Quantum utilities"], "utils/utils", "dynamiqs/utils/utils.py"),
    (
        ["Utilities", "Converting tensors"],
        "utils/tensor_types",
        "dynamiqs/utils/tensor_types.py",
    ),
    (
        ["Utilities", "Wigner representation"],
        "utils/wigners",
        "dynamiqs/utils/wigners.py",
    ),
]


def parse_dunder_all(file_path):
    """Parse a file to find all elements of the __all__ attribute."""
    all_functions = []
    in_all_block = False

    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line in lines:
            if '__all__' in line:
                in_all_block = True
                start_index = line.index('__all__')
                line = line[start_index:]

            if in_all_block:
                if '[' in line:
                    start_index = line.index('[')
                if ']' in line:
                    end_index = line.index(']')
                    in_all_block = False

                    all_list = line[start_index + 1 : end_index].split(',')
                elif '__all__' in line:
                    all_list = line[start_index + 1 :].split(',')
                else:
                    all_list = line.split(',')

                all_list = [function.strip().strip("'") for function in all_list]
                all_list = [function for function in all_list if function != '']
                all_functions.extend([function for function in all_list])

    return all_functions


for doc_path, ref_path, src_path in FILES_TO_PARSE:
    # convert various paths
    doc_path = Path(doc_path)
    ref_path = Path(ref_path)
    src_path = Path(src_path)
    full_ref_path = Path("reference", ref_path.with_suffix(".md"))

    # get global navigation path
    parts = list(doc_path.parts)

    # get global src identifier
    identifier = ".".join(list(src_path.with_suffix("").parts))

    # loop over all functions in file
    for function in parse_dunder_all(src_path):
        ref_path_fn = Path(ref_path, function)
        full_ref_path_fn = Path("reference", ref_path_fn.with_suffix(".md"))

        # add function page to navigation
        nav[parts + [function]] = ref_path_fn.with_suffix(".md").as_posix()

        # create the function page
        with mkdocs_gen_files.open(full_ref_path_fn, "w") as fd:
            print(f"::: {identifier}.{function}", file=fd)
            print("    options:", file=fd)
            print("        show_root_heading: true", file=fd)
            print("        heading_level: 1", file=fd)

        mkdocs_gen_files.set_edit_path(full_ref_path_fn, Path("../") / src_path)

# make the navigation file
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
