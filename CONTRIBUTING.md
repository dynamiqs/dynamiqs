# Contributing to Dynamiqs

## Requirements

The project was written using Python 3.9+, you must have a compatible version of Python (i.e. >= 3.9) installed on your computer.

## Setup

Clone the repository and dive in:

```shell
git clone git@github.com:dynamiqs/dynamiqs.git
cd dynamiqs
```

We strongly recommend that you create a virtual environment to install the project dependencies. You can then install the library (in editable mode) with all its dependencies:

```shell
pip install -e .
```

You also need to install the developer dependencies:

```shell
pip install -e ".[dev]"
```

## Code style

This project follows PEP8 and uses automatic formatting and linting tools to ensure that the code is compliant.

The maximum line length is **88**, we recommend that you set this limit in your IDE.

## Workflow

### Before submitting a pull request (run all tasks)

Run all tasks before each commit:

```shell
task all
```

### Run some tasks automatically before each commit

Alternatively, you can use `pre-commit` to automatically run the cleaning tasks (ruff + codespell) before each commit:

```shell
pip install pre-commit
pre-commit install
```

### Build the documentation

The documentation is built using [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. MkDocs generates a static website based on the markdown files in the `docs/` directory.

To preview the changes to the documentation as you edit the docstrings or the markdown files in `docs/`, we recommend starting a live preview server, which will automatically rebuild the website upon modifications:

```shell
task docserve
```

Open <http://localhost:8000/> in your web browser to preview the documentation website.

You can build the static documentation website locally with:

```shell
task docbuild
```

This will create a `site/` directory with the contents of the documentation website. You can then simply open `site/index.html` in your web browser to view the documentation website.

### Run specific tasks

You can also execute tasks individually:

```shell
> task --list
lint         lint the code (ruff)
format       auto-format the code (ruff)
codespell    check for misspellings (codespell)
clean        clean the code (ruff + codespell)
test         run the unit tests suite (pytest)
doctest-code check code docstrings examples (doctest)
doctest-docs check documentation examples (doctest)
doctest      check all examples (doctest)
docbuild     build the documentation website
docserve     preview documentation website with hot-reloading
all          run all tasks before a commit (ruff + codespell + pytest + doctest)
ci           run all the CI checks
```
