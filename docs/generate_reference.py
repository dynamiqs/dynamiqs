"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

# get navigation dictionary
nav = mkdocs_gen_files.Nav()

# static page
nav[["Utilities"]] = "utils.md"

# files to parse
FILES_TO_PARSE = [
    # (Path in navigation, path to source)
    (["Utilities", "Operators"], "utils/operators"),
    (["Utilities", "Quantum states"], "utils/states"),
    (["Utilities", "Quantum utilities"], "utils/utils"),
    (["Utilities", "Converting tensors"], "utils/tensor_types"),
    (["Utilities", "Wigner representation"], "utils/wigners"),
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


for nav_path, path in FILES_TO_PARSE:
    # convert various paths
    path = Path(path)
    src_path = Path("dynamiqs", path.with_suffix(".py"))
    ref_path = Path("reference", path.with_suffix(".md"))

    # get global src identifier
    identifier = ".".join(list(src_path.with_suffix("").parts))

    # loop over all functions in file
    for function in parse_dunder_all(src_path):
        path_fn = Path(path, function)
        ref_path_fn = Path("reference", path_fn.with_suffix(".md"))

        # add function page to navigation
        nav[nav_path + [function]] = path_fn.with_suffix(".md").as_posix()

        # create the function page
        with mkdocs_gen_files.open(ref_path_fn, "w") as fd:
            print(f"::: {identifier}.{function}", file=fd)
            print("    options:", file=fd)
            print("        show_root_heading: true", file=fd)
            print("        heading_level: 1", file=fd)

        mkdocs_gen_files.set_edit_path(ref_path_fn, Path("../") / src_path)

# write to the navigation file
with mkdocs_gen_files.open("reference/navigation.md", "a") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
