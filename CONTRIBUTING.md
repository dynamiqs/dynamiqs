# Contributing to torchqdynamics

## Requirements

The project was written using Python 3.8+, you must have a compatible version of Python (i.e. >= 3.8) installed on your computer.

## Setup

Clone the repository and dive in:

```shell
git clone git@github.com:PierreGuilmin/torchqdynamics.git
cd torchqdynamics
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

Alternatively, you can use `pre-commit` to automatically run the linting tasks (isort + black) before each commit:

```shell
pip install pre-commit
pre-commit install
```

### Run specific tasks

You can also execute tasks individually:

```shell
task --list
isort     sort the imports (isort)
black     auto-format the code (black)
codespell check for misspellings (codespell)
lint      lint the code (isort + black + codespell)
test      run the unit tests suite (pytest)
all       run all tasks before a commit (isort + black + codespell + pytest)
ci        run all the CI checks
```
