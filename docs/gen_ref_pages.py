"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files


def find_all_functions(file_path):
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


nav = mkdocs_gen_files.Nav()

for path in sorted(Path("dynamiqs/utils").rglob("*.py")):  #
    module_path = path.relative_to("").with_suffix("")  #
    doc_path = path.relative_to("").with_suffix(".md")  #
    full_doc_path = Path("reference", doc_path)  #

    parts = list(module_path.parts)

    if parts[-1] == "__init__":  #
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    nav[parts] = doc_path.as_posix()  #
    identifier = ".".join(parts)  #

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  #
        print(f"::: {identifier}", file=fd)  #
        print("    options:", file=fd)
        print("        table: true", file=fd)

    mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)  #

    for function in find_all_functions(path.with_suffix(".py")):
        doc_path = Path(path.relative_to(""), function).with_suffix(".md")  #
        full_doc_path = Path("reference", doc_path)

        nav[parts + [function]] = doc_path.as_posix()  #

        with mkdocs_gen_files.open(full_doc_path, "w") as fd:  #
            print(f"::: {identifier}.{function}", file=fd)  #
            print("    options:", file=fd)
            print("        show_root_heading: true", file=fd)
            print("        heading_level: 1", file=fd)

        mkdocs_gen_files.set_edit_path(full_doc_path, Path("../") / path)  #

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  #
    nav_file.writelines(nav.build_literate_nav())  #
