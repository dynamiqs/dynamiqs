# Contributing to Dynamiqs

We warmly welcome all contributions. If you're a junior developer or physicist, you can start with a small utility function, and move on to bigger problems as you discover the library's internals. If you're more experienced and want to implement more advanced features, don't hesitate to [get in touch](https://github.com/dynamiqs/dynamiqs#lets-talk) to discuss what would suit you.

To contribute efficiently, a few guidelines are compiled below.

## 1. Requirements

The project was written using Python 3.10+, you must have a compatible version of Python (i.e. >= 3.10) installed on your computer.

## 2. Setup

Clone the repository and dive in:

```shell
git clone git@github.com:dynamiqs/dynamiqs.git
cd dynamiqs
```

To install the library with all its dependencies, as well as the developer dependencies:

- **If you use `uv`**, just run:
  ```shell
  uv sync --extra dev
  ```
- **If you use `pip`**, we strongly recommend creating a virtual environment to install the project dependencies. You can follow [this guide](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/) to set one up. You can then install the library in editable mode:
  ```shell
  pip install -e ".[dev]"
  ```

## 3. Workflow

### Github flow

We use the GitHub flow branch-based workflow to collaborate on the library. If you're unfamiliar with open-source development, you can follow the [GitHub flow tutorial](https://docs.github.com/en/get-started/using-github/github-flow). Briefly, to add a change you should: fork the repository, create a branch, make and commit your changes, create a pull request (PR), address review comments, and merge the PR.

### Planning your change

If you are planning a large change (more than one hundred lines of code), it is usually more efficient to discuss it with us beforehand. This way, we can agree on the API and the main logic. For structural or significant changes, we typically write and iterate on small RFCs (Requests for Comments). For large change, please separate the logical building blocks of your change into distinct pull requests as much as possible. Long PRs are harder to review, and breaking them down will make everyone's life easier.

If the proposed change is necessary for your work and has not been reviewed within reasonable time (a few working days), don’t hesitate to ping the main developers directly on the PR.

### Developer tools

We use a variety of modern formatting and linting tools, as well as automated tests, to ensure high code quality and to identify bugs early in the development process. These tests can be run either together or individually using the `task` CLI tool, which is installed as part of the development dependencies.

Typically you should ensure all tasks run smoothly before submitting a new PR:

```shell
task all
```

Here is a list of available tasks:

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

### Run tasks automatically before each commit

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

## 4. Style guide

This project adheres to PEP 8 guidelines. The maximum line length is **88** characters, and we recommend setting this limit in your IDE. Generally, our automatic cleaning task (`task clean`) will address most basic styling issues. If you have any remaining doubts, you can follow the conventions used throughout the library by looking at various files.

We strive to keep the codebase as maintainable, readable, and unified as possible. We would appreciate your help by following these guidelines in your next PR.

### Writing a PR

The main goal to keep in mind is that if someone reads a past PR, they should quickly gain a clear understanding of what the change was.

- Use a lower case name for your branch name.
- Choose a clear and precise title for your PR. We use the squash and merge strategy to incorporate changes, so the title of your PR will become the commit title in the `main` branch history. Refer to the [commit history of the `main` branch](https://github.com/dynamiqs/dynamiqs/commits/main/) for inspiration.
- If, after multiple reviews and iterations, the content of the PR no longer corresponds to the original description, please update the description.

### Adding a new function

When writing your function:

- Type all arguments of your function, and its return type.
- When typing a public function, use `QArrayLike` for qarray inputs, `asqarray()` to convert to a proper qarray, and `QArray` for qarray outputs. Use `ArrayLike` for array inputs, `jnp.asarray()` to convert to a proper array, and `Array` for array outputs. See related [JAX typing best practices](https://jax.readthedocs.io/en/latest/jax.typing.html#jax-typing-best-practices).
- Ensure you run `task docserve` to verify how the documentation for your function appears, and correct any issues with rendering.
- To type a qarray or an array shape, use `...` to denote possible batch dimensions, use `n` for the Hilbert space dimension.
- To add an axis to a qarray or an array, favor `[None, ...]` over `unsqueeze`.
- Favor `(...).sum(0)` over `jnp.sum(..., 0)`.

Make sure to add your function:

1. to the `__all__` variable at the top of the file (this is how we control namespaces to have everything available under `dq.*`),
2. to the `dynamiqs/mkdocs.yml` file (so it appears in the documentation and navigation),
3. to the `dynamiqs/docs/python_api/index.md` file (so it appears on the Python API page, which lists all the functions of the library).

### Writing a docstring

We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#s3.8.1-comments-in-doc-strings) (see examples [here](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)) with a few quirks.

- Headers can include (in this order and with these names): `Args`, `Returns`, `Raises`, `Examples`, `See also`.
- You can use the admonitions (colored blocks in the documentation) `Note` and `Warning` if relevant.
- Avoid using `The` in arguments description, for example change `x: The quantum state.` to `x: Quantum state.`.
- Specify arguments type in `_(...)_` after the argument name and _only if it is necessary_. Adding a type can for example be useful to add a shape information for a qarray or an array, or because the argument typing in the function signature is opaque.

### Documentation

- For internal links, use `[dq.Options][dynamiqs.Options]` for a class, `[dq.sesolve()][dynamiqs.sesolve]` for a function (explicitely include `()` in the function name), and `(doc page)(relative/path/to/file.md)` for another documentation page (note the parentheses instead of brackets).
- If you want to add an icon somewhere, use one of the `:material-*` (search the list [here](https://squidfunk.github.io/mkdocs-material/reference/icons-emojis/)).

### Exceptions message

- Use one or multiple sentences starting with a capital letter and ending with a period.
- Use backticks ``` ` ``` to refer to a variable name or a function. Use `'` to refer to a string.
- Many exceptions arise from incorrect argument input; they should typically follow the format: `"Argument ... must ..., but ..."`.

Examples:

```python
raise ValueError(f'Argument `H` must have shape (n, n), but has shape H.shape={H.shape}.')
raise ValueError(
    "Argument `matmul_precision` should be a string 'low', 'high', or"
    f" 'highest', but is '{matmul_precision}'."
)
```
